import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, EncoderDecoderModel, EncoderDecoderConfig, BertConfig, AdamW
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Sequential
import torch.nn as nn
import torch.nn.functional as F
import json
from tqdm import tqdm
import numpy as np

# Load data from the JSON files
with open('TaskInfo_ObjectGraph.json', 'r') as file:
    task_data = json.load(file)

with open('GAT_embedding_Dim768.json', 'r') as file:
    object_data = json.load(file)

# Prepare object feature mapping
object_features = {}
for obj in object_data:
    object_features[obj['object1']] = obj['properties1']
    object_features[obj['object2']] = obj['properties2']

# Custom Dataset with integrated object features
class TaskDataset(Dataset):
    def __init__(self, data, object_features, tokenizer, max_length=512):
        self.data = data
        self.object_features = object_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize input and output texts
        inputs = self.tokenizer(item['task_desc'] + ' ' + item['high_descs'],
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        outputs = self.tokenizer(item['actions'],
                                 padding='max_length',
                                 truncation=True,
                                 max_length=self.max_length,
                                 return_tensors="pt")

        # Retrieve or compute object features
        object_feature = np.zeros(self.max_length)  # Initialize with zeros
        if 'nodes_info' in item and len(item['nodes_info']) == 1:
            node_id = item['nodes_info'][0]['id']
            obj_name = [obj['object1'] for obj in object_data if obj['object1'] == node_id or obj['object2'] == node_id]
            if obj_name:
                object_feature = self.object_features.get(obj_name[0], object_feature)
        elif 'nodes_info' in item and 'links_info' in item:
            # Prepare edges for GAT
            node_indices = {node['id']: i for i, node in enumerate(item['nodes_info'])}
            edge_index = []
            for link in item['links_info']:
                source = node_indices[link['source']]
                target = node_indices[link['target']]
                edge_index.append([source, target])
                if not item.get('directed', True):
                    edge_index.append([target, source])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Dummy features for GAT nodes
            num_nodes = len(item['nodes_info'])
            features = torch.eye(num_nodes)
            data = Data(x=features, edge_index=edge_index)

            # Use a simple GAT model to encode features
            model = GATWithSingleHead(num_nodes, self.max_length)
            gat_features = model.encode(data.x, data.edge_index).detach().numpy()
            object_feature = np.mean(gat_features, axis=0)

        # Ensure object_feature is of length max_length
        if len(object_feature) < self.max_length:
            object_feature = np.pad(object_feature, (0, self.max_length - len(object_feature)), mode='constant')
        else:
            object_feature = object_feature[:self.max_length]
            object_feature = object_feature[:self.max_length]

        # Concatenate object features with tokenized inputs
        input_ids = inputs.input_ids.squeeze(0)
        input_ids = torch.cat((input_ids, torch.tensor(object_feature, dtype=torch.long)[:self.max_length]), dim=0)[
                    :self.max_length]

        return input_ids, outputs.input_ids.squeeze(0)


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Define GAT model
class GATWithSingleHead(nn.Module):
    def __init__(self, num_features, out_features):
        super(GATWithSingleHead, self).__init__()
        self.encoder = Sequential('x, edge_index', [
            (GATConv(num_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(out_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(out_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATConv(out_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
        ])

    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return torch.tanh(x)

    def decode(self, z):
        adj = torch.sigmoid(torch.matmul(z, z.t()))
        return adj

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        adj_recon = self.decode(z)  # 使用 decode 方法来计算重建的邻接矩阵
        return adj_recon, z


# Define configurations for both encoder and decoder with modified dropout
encoder_config = BertConfig.from_pretrained('bert-base-uncased', num_hidden_layers=6, hidden_dropout_prob=0.3)
decoder_config = BertConfig.from_pretrained('bert-base-uncased', num_hidden_layers=6, hidden_dropout_prob=0.3)

# Create the combined model configuration
model_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
model = EncoderDecoderModel(config=model_config)

# Set decoder_start_token_id, pad_token_id, and eos_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id

# Create the dataset and dataloader
dataset = TaskDataset(task_data, object_features, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training parameters with weight decay
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
model.train()

# Define early stopping criteria
min_loss = float('inf')
patience = 2
trigger_times = 0

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
        input_ids, labels = batch
        attention_mask = (input_ids != tokenizer.pad_token_id).type(torch.LongTensor)
        decoder_attention_mask = (labels != tokenizer.pad_token_id).type(torch.LongTensor)

        input_ids, labels = input_ids.to(device), labels.to(device)
        attention_mask, decoder_attention_mask = attention_mask.to(device), decoder_attention_mask.to(device)

        outputs = model(input_ids=input_ids, labels=labels,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(loader)
    print(f"Average Loss: {average_epoch_loss}")

    # Early stopping
    if average_epoch_loss < min_loss:
        min_loss = average_epoch_loss
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break

# Save the trained model
model.save_pretrained('bert_gat_model')
