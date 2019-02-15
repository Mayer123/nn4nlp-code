import torch

class CNNSimpleAtt(torch.nn.Module):
    def __init__(self, nwords, embeddings, emb_size, num_filters, ntags):
        super(CNNSimpleAtt, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        self.embedding.weight = torch.nn.Parameter(torch.as_tensor(embeddings, dtype=torch.float32))
        # Conv 1d
        self.conv = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=1)
        self.conv0 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=3)
        self.conv1 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=4)
        self.conv2 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=5)
        self.relu = torch.nn.ReLU()
        self.embed_dropout = torch.nn.Dropout()
        self.dropout = torch.nn.Dropout()
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = torch.nn.MaxPool1d(num_filters)

        self.projection_layer = torch.nn.Linear(in_features=num_filters*4, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        
    def forward(self, words, masks):
        emb = self.embedding(words)                 # batch x nwords x emb_size
        
        emb = self.embed_dropout(emb)
        emb = emb.permute(0, 2, 1)                  # 1 x emb_size x nwords
        h = self.conv(emb)
        h = self.relu(h)
        h0 = self.conv0(emb)                        # batch x num_filters x nwords
        h0 = self.relu(h0)
        h1 = self.conv1(emb)
        h1 = self.relu(h1)
        h2 = self.conv2(emb)
        h2 = self.relu(h2)
        
        pool = self.pool(h.permute(0, 2, 1))
        masks = torch.unsqueeze(masks, dim=-1)
        pool = pool * masks
        att = self.softmax(pool)

        vec = torch.bmm(h, att)
        vec = torch.squeeze(vec)

        # Do max pooling
        h0 = h0.max(dim=2)[0]                         # 1 x num_filters
        h1 = h1.max(dim=2)[0]
        h2 = h2.max(dim=2)[0]
              
        out = torch.cat([h0, h1, h2, vec], dim=1)
        out = self.dropout(out)
        out = self.projection_layer(out)              # size(out) = 1 x ntags
        return out

    def evaluate(self, words, masks):
        emb = self.embedding(words)                 # batch x nwords x emb_size
        
        emb = emb.permute(0, 2, 1)                  # 1 x emb_size x nwords
        h = self.conv(emb)
        h = self.relu(h)
        h0 = self.conv0(emb)                        # batch x num_filters x nwords
        h0 = self.relu(h0)
        h1 = self.conv1(emb)
        h1 = self.relu(h1)
        h2 = self.conv2(emb)
        h2 = self.relu(h2)

        pool = self.pool(h.permute(0, 2, 1))
        masks = torch.unsqueeze(masks, dim=-1)
        pool = pool * masks
        
        att = self.softmax(pool)
        vec = torch.bmm(h, att)
        vec = torch.squeeze(vec)

        # Do max pooling
        h0 = h0.max(dim=2)[0]                         # 1 x num_filters
        h1 = h1.max(dim=2)[0]
        h2 = h2.max(dim=2)[0]

        out = torch.cat([h0, h1, h2, vec], dim=1)
        out = self.projection_layer(out)              # size(out) = 1 x ntags
        return out

class CNNclass(torch.nn.Module):
    def __init__(self, nwords, embeddings, emb_size, num_filters, ntags):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        self.embedding.weight = torch.nn.Parameter(torch.as_tensor(embeddings, dtype=torch.float32))

        #self.conv = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=1)

        self.conv0 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=3)
        self.conv1 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=4)
        self.conv2 = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=5)
        self.relu = torch.nn.ReLU()
        self.embed_dropout = torch.nn.Dropout()
        self.dropout = torch.nn.Dropout()
        #self.w_qs = torch.nn.Linear(in_features=emb_size, out_features=100)
        self.projection_layer = torch.nn.Linear(in_features=num_filters*3, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)
        #torch.nn.init.xavier_uniform_(self.w_qs.weight)

    def forward(self, words, masks):
        emb = self.embedding(words)                 # batch x nwords x emb_size
        
        emb = self.embed_dropout(emb)
        emb = emb.permute(0, 2, 1)                  # 1 x emb_size x nwords
        
        h0 = self.conv0(emb)                        # batch x num_filters x nwords
        h0 = self.relu(h0)
        h1 = self.conv1(emb)
        h1 = self.relu(h1)
        h2 = self.conv2(emb)
        h2 = self.relu(h2)
        # Do max pooling
        h0 = h0.max(dim=2)[0]                         # 1 x num_filters
        h1 = h1.max(dim=2)[0]
        h2 = h2.max(dim=2)[0]
              
        out = torch.cat([h0, h1, h2], dim=1)
        out = self.dropout(out)
        out = self.projection_layer(out)              # size(out) = 1 x ntags
        return out

    def evaluate(self, words, masks):
        emb = self.embedding(words)                 # batch x nwords x emb_size
        emb = emb.permute(0, 2, 1)                  # 1 x emb_size x nwords
        h0 = self.conv0(emb)                        # batch x num_filters x nwords
        h0 = self.relu(h0)
        h1 = self.conv1(emb)
        h1 = self.relu(h1)
        h2 = self.conv2(emb)
        h2 = self.relu(h2)
        # Do max pooling
        h0 = h0.max(dim=2)[0]                         # 1 x num_filters
        h1 = h1.max(dim=2)[0]
        h2 = h2.max(dim=2)[0]

        out = torch.cat([h0, h1, h2], dim=1)
        out = self.projection_layer(out)              # size(out) = 1 x ntags
        return out
