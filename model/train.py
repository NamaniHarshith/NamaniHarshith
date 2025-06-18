import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def move_batch_to_device(seqs, lengths, device):
    seqs = [seq.to(device) for seq in seqs]
    lengths = [length.to(device) for length in lengths]
    return seqs, lengths

class HospitalDataset(Dataset):
    def __init__(self, data, max_len=50):
        self.max_len = max_len
        # Lowercase everything for vocab building
        self.char_vocab = self._build_vocab(data)
        self.precomp = []
        for rec in data:
            entry = {}
            for f in ['state','district','city','name','address']:
                text = rec[f].lower()  # Lowercasing here
                seq = [self.char_vocab.get(c, 1) for c in text[:max_len]]
                length = len(seq)
                seq += [0] * (max_len - length)
                entry[f] = torch.tensor(seq, dtype=torch.long)
                entry[f + '_len'] = length
            entry['meta'] = rec
            self.precomp.append(entry)

    def _build_vocab(self, data):
        chars = set(''.join(rec[f].lower() for rec in data for f in ['state','district','city','name','address']))
        vocab = {c: i+2 for i, c in enumerate(sorted(chars))}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    def __len__(self):
        return len(self.precomp)

    def __getitem__(self, idx):
        entry = self.precomp[idx]
        orig = []
        mod  = []
        for f in ['state','district','city','name','address']:
            seq = entry[f].clone()
            length = entry[f + '_len']
            seq_mod = seq.clone()
            if random.random() < 0.3 and length > 0:
                aug_type = random.choice(['replace', 'swap', 'delete'])
                if aug_type == 'replace':
                    pos = random.randrange(length)
                    seq_mod[pos] = random.choice(list(self.char_vocab.values()))
                elif aug_type == 'swap' and length >= 2:
                    pos1, pos2 = random.sample(range(length), 2)
                    seq_mod[pos1], seq_mod[pos2] = seq_mod[pos2], seq_mod[pos1]
                elif aug_type == 'delete':
                    pos = random.randrange(length)
                    seq_mod[pos] = 0  # set to pad
            orig.extend([seq, length])
            mod.extend([seq_mod, length])
        return orig, mod


def collate_fn(batch):
    orig_batch = [[] for _ in range(10)]
    mod_batch  = [[] for _ in range(10)]
    for orig, mod in batch:
        for i in range(10):
            orig_batch[i].append(orig[i])
            mod_batch[i].append(mod[i])

    def stack(x_list, is_len):
        if is_len:
            return torch.tensor(x_list, dtype=torch.long)
        return torch.stack(x_list)

    orig_stacked = [stack(orig_batch[i], i % 2 == 1) for i in range(10)]
    mod_stacked  = [stack(mod_batch[i], i % 2 == 1)  for i in range(10)]
    return orig_stacked, mod_stacked

class Embedder(nn.Module):
    def __init__(self, nvocab, emb_dim=128, hid=128):
        super().__init__()
        self.char_emb = nn.Embedding(nvocab, emb_dim, padding_idx=0)
        self.lstm = nn.ModuleDict({
            f: nn.LSTM(emb_dim, hid, num_layers=2, batch_first=True)
            for f in ['s','di','c','n','a']
        })
        self.attn = nn.ModuleDict({
            f: nn.Linear(hid, 1)
            for f in ['s','di','c','n','a']
        })
        self.proj = nn.ModuleDict({
            f: nn.Linear(hid, emb_dim)
            for f in ['s','di','c','n','a']
        })
        self.dropout = nn.Dropout(0.2)

    def forward(self, seqs, lengths):
        outputs = {}
        for f, seq, length in zip(['s','di','c','n','a'], seqs, lengths):
            emb = self.char_emb(seq)
            packed = pack_padded_sequence(emb, length.cpu(), batch_first=True, enforce_sorted=False)
            output, (h, c) = self.lstm[f](packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
            scores = self.attn[f](output).squeeze(-1)
            # Mask padding
            seq_len = output.size(1)
            mask = torch.arange(seq_len, device=output.device).unsqueeze(0).expand(seq.size(0), seq_len) < length.unsqueeze(1)
            scores[~mask] = -1e9
            attn_weights = nn.functional.softmax(scores, dim=1)
            context = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)
            vec = self.proj[f](context)
            outputs[f] = nn.functional.normalize(self.dropout(vec), dim=-1)
        return outputs

def train(emb_dim=128, hid=128):
    os.makedirs('output', exist_ok=True)
    data = json.load(open('existing_formatted_hospitals.json','r',encoding='utf-8'))
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    dataset = HospitalDataset(data)

    train_loader = DataLoader(dataset, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(dataset, batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = Embedder(nvocab=len(dataset.char_vocab), emb_dim=emb_dim, hid=hid).to(device)
    criterion = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val = float('inf')
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for orig, mod in train_loader:
            seqs_o, lengths_o = orig[0::2], orig[1::2]
            seqs_m, lengths_m = mod[0::2], mod[1::2]
            seqs_o, lengths_o = move_batch_to_device(seqs_o, lengths_o, device)
            seqs_m, lengths_m = move_batch_to_device(seqs_m, lengths_m, device)

            optimizer.zero_grad()
            emb_o = model(seqs_o, lengths_o)
            emb_m = model(seqs_m, lengths_m)
            batch_size = emb_o['s'].size(0)
            pos_target = torch.ones(batch_size, device=device)
            neg_target = -torch.ones(batch_size, device=device)

            loss = 0
            for f in emb_o:
                sim = torch.matmul(emb_o[f], emb_o[f].T)
                mask = torch.eye(batch_size, device=device) * -1e9
                sim = sim + mask
                hard_neg_indices = torch.argmax(sim, dim=1)
                hard_neg = emb_o[f][hard_neg_indices]
                loss += criterion(emb_o[f], emb_m[f], pos_target) + criterion(emb_o[f], hard_neg, neg_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        total_cos_sim = 0
        with torch.no_grad():
            for orig, mod in val_loader:
                seqs_o, lengths_o = orig[0::2], orig[1::2]
                seqs_m, lengths_m = mod[0::2], mod[1::2]
                seqs_o, lengths_o = move_batch_to_device(seqs_o, lengths_o, device)
                seqs_m, lengths_m = move_batch_to_device(seqs_m, lengths_m, device)
                emb_o = model(seqs_o, lengths_o)
                emb_m = model(seqs_m, lengths_m)
                batch_size = emb_o['s'].size(0)
                pos_target = torch.ones(batch_size, device=device)
                val_loss += sum(criterion(emb_o[f], emb_m[f], pos_target).item() for f in emb_o)
                cos_sim_sum = sum((emb_o[f] * emb_m[f]).sum(dim=1).mean().item() for f in emb_o)
                total_cos_sim += cos_sim_sum

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_cos_sim = total_cos_sim / (len(val_loader) * len(emb_o))
        print(f"Epoch {epoch} ▶ train_loss={avg_train:.4f} val_loss={avg_val:.4f} avg_cos_sim={avg_cos_sim:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'model_state': model.state_dict(),
                'char_vocab': dataset.char_vocab
            }, 'output/best_model_highdim.pt')
        scheduler.step(avg_val)

    # Generate and save component vectors with original texts
    checkpoint = torch.load('output/best_model_highdim.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    comps = []
    for entry in dataset.precomp:
        rec = entry['meta']
        record = {
            'id': rec.get('id', None),
            'state': rec['state'],
            'district': rec['district'],
            'city': rec['city'],
            'name': rec['name'],
            'address': rec['address']
        }
        seqs = [entry[f] for f in ['state','district','city','name','address']]
        lengths = [entry[f + '_len'] for f in ['state','district','city','name','address']]
        seqs = [x.unsqueeze(0).to(device) for x in seqs]
        lengths = [torch.tensor([l], device=device) for l in lengths]
        with torch.no_grad():
            emb = model(seqs, lengths)
        for f, key in zip(['s','di','c','n','a'], ['state_vector','district_vector','city_vector','name_vector','address_vector']):
            record[key] = emb[f].squeeze(0).cpu().tolist()
        comps.append(record)
    json.dump(comps, open('output/component_vectors_highdim.json','w'), indent=2)

if __name__ == '__main__':
    train(emb_dim=256, hid=256)
