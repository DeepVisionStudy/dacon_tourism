import torch
import torch.nn as nn
from transformers import AutoModel


class TourClassifier(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name, device):
        super(TourClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.image_model = AutoModel.from_pretrained(image_model_name).to(device)
        
        self.text_model.gradient_checkpointing_enable()
        self.image_model.gradient_checkpointing_enable()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        self.drop = nn.Dropout(p=0.1)

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
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, target_size),
        )