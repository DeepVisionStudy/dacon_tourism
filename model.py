import torch
import torch.nn as nn
from transformers import AutoModel


class TourClassifier(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name, device, dropout):
        super(TourClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.image_model = AutoModel.from_pretrained(image_model_name).to(device)
        
        self.text_model.gradient_checkpointing_enable()
        self.image_model.gradient_checkpointing_enable()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        self.dropout_ratio = dropout
        self.drop = nn.Dropout(p=dropout)

        self.cls = self._get_cls(n_classes1)
        self.cls2 = self._get_cls(n_classes2)
        self.cls3 = self._get_cls(n_classes3)
    
    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_output = self.image_model(pixel_values=pixel_values)
        concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state], 1)
        
        outputs = self.transformer_encoder(concat_outputs)
        #cls token 
        outputs = outputs[:,0]
        output = self.drop(outputs)

        out1 = self.cls(output)
        out2 = self.cls2(output)
        out3 = self.cls3(output)
        return out1, out2, out3
    
    def _get_cls(self, target_size):
        return nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
            nn.LayerNorm(self.text_model.config.hidden_size),
            nn.Dropout(p=self.dropout_ratio),
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, target_size),
        )


class TourClassifier_Continuous(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name, device, dropout):
        super(TourClassifier_Continuous, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.image_model = AutoModel.from_pretrained(image_model_name).to(device)
        
        self.text_model.gradient_checkpointing_enable()
        self.image_model.gradient_checkpointing_enable()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        self.dropout_ratio = dropout
        self.drop = nn.Dropout(p=dropout)

        self.body = self._get_body()
        self.body2 = self._get_body()
        self.body3 = self._get_body()

        self.head = nn.Linear(self.text_model.config.hidden_size, n_classes1)
        self.head2 = nn.Linear(self.text_model.config.hidden_size, n_classes2)
        self.head3 = nn.Linear(self.text_model.config.hidden_size, n_classes3)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_output = self.image_model(pixel_values=pixel_values)
        concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state], 1)
        
        outputs = self.transformer_encoder(concat_outputs)
        #cls token 
        outputs = outputs[:,0]
        output = self.drop(outputs)

        output = self.body(output)
        out1 = self.head(output)
        output = self.body2(output)
        out2 = self.head2(output)
        output = self.body3(output)
        out3 = self.head3(output)
        return out1, out2, out3
    
    def _get_body(self):
        return nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
            nn.LayerNorm(self.text_model.config.hidden_size),
            nn.Dropout(p=self.dropout_ratio),
            nn.ReLU(),
        )