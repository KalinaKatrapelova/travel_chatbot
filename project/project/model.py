import torch
import torch.nn as nn


class EnhancedNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, embedding_dim=100, dropout_rate=0.2):
        super(EnhancedNeuralNet, self).__init__()

        # Define model architecture for word embeddings input
        self.embedding_dim = embedding_dim

        # First hidden layer
        self.l1 = nn.Linear(embedding_dim, hidden_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third hidden layer with reduced size
        self.l3 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.l4 = nn.Linear(hidden_size // 2, num_classes)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)

        # LSTM layer for sequence processing (not used in the default forward method)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, x):
        # Check if we're dealing with a single sample (for inference)
        is_single_sample = (x.dim() == 1 or (x.dim() == 2 and x.size(0) == 1))
        if is_single_sample and x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        # First layer
        out = self.l1(x)
        if out.dim() > 1 and out.size(0) > 1:  # Only apply batch norm for batch size > 1
            out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.dropout1(out)

        # Second layer
        out = self.l2(out)
        if out.dim() > 1 and out.size(0) > 1:
            out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.dropout2(out)

        # Third layer
        out = self.l3(out)
        if out.dim() > 1 and out.size(0) > 1:
            out = self.bn3(out)
        out = self.leaky_relu(out)
        out = self.dropout3(out)

        # Output layer
        out = self.l4(out)

        if is_single_sample and out.size(0) == 1:
            out = out.squeeze(0)  # Remove batch dimension if it was a single sample

        return out

    def forward_with_lstm(self, x, sequence_lengths=None):
        """Alternative forward method using LSTM (can be used for sequence data)"""
        # x shape: (batch_size, sequence_length, embedding_dim)

        if sequence_lengths is not None:
            # Pack padded sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sequence_lengths, batch_first=True, enforce_sorted=False
            )
            # Process with LSTM
            packed_out, (hidden, _) = self.lstm(packed_x)
            # Unpack sequence
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            # Use the last hidden state
            out = hidden[-1]
        else:
            # If no sequence lengths provided, use the last hidden state
            _, (hidden, _) = self.lstm(x)
            out = hidden[-1]

        # Continue with regular forward pass
        out = self.l1(out)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.dropout1(out)

        out = self.l2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.dropout2(out)

        out = self.l3(out)
        out = self.bn3(out)
        out = self.leaky_relu(out)
        out = self.dropout3(out)

        out = self.l4(out)

        return out


# LSTM-based model variant
class LSTMNeuralNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout_rate=0.2):
        super(LSTMNeuralNet, self).__init__()

        # Word embedding layer (for sequence input)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, seq_lengths):
        # x shape: (batch_size, max_seq_length)

        # Get embeddings
        embedded = self.embedding(x)
        # embedded shape: (batch_size, max_seq_length, embedding_dim)

        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lengths, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch_size, max_seq_length, hidden_size*2)

        # Attention mechanism
        attention_scores = self.attention(output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)

        # Pass through fully connected layers
        out = self.fc1(context_vector)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out