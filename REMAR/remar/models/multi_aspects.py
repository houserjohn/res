import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    ExponentialLR

from remar.vocabulary import Vocabulary
from remar.vocabulary import UNK_TOKEN, PAD_TOKEN

from remar.common.util import get_encoder
from remar.nn.lstm_encoder import LSTMEncoder

from remar.models.latent import LatentRationaleModel
from remar.common.classifier import Rating_Classifier, Polarity_Classifier, Testing_Classifier
from remar.utils import get_args, load_embeddings, amazon_reader, initialize_model_, \
                        prepare_minibatch, get_minibatch, print_parameters, pad

from remar.models.transformer import TransformerModel

cfg = get_args()
cfg = vars(cfg)

device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu')
print("device:", device)

class MultiAspectsLatentRationaleModel(nn.Module):
    """
    Generate a latent model for each aspect
    """

    def __init__(self,
                aspects,
                cfg=None,
                vocab=None,
                ):

        super(MultiAspectsLatentRationaleModel, self).__init__()
        self.aspects = aspects
        self.latent_models = []
        self.transformers = []
        self.aspect_rating_predictors = []
        self.aspect_rating_classifiers = []
        self.aspect_polarity_predictors = []
        self.aspect_polarity_classifiers = []


        self.vocab = vocab
        vectors = load_embeddings(cfg["embeddings"], vocab)
    
        output_size = 1
        emb_size = cfg["emb_size"]
        hidden_size = cfg["hidden_size"]
        dropout = cfg["dropout"]
        layer = cfg["layer"]
        vocab_size = len(vocab.w2i)
        dependent_z = cfg["dependent_z"]
        selection = cfg["selection"]
        lasso = cfg["lasso"]
        lagrange_alpha = cfg["lagrange_alpha"]
        lagrange_lr = cfg["lagrange_lr"]
        lambda_init = cfg["lambda_init"]
        lambda_min = cfg["lambda_min"]
        lambda_max = cfg["lambda_max"]

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.L2_regularize = cfg["L2_regularize"]
        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = get_encoder(layer, emb_size, hidden_size)


        for i in range(self.aspects):
            latent_model = LatentRationaleModel(
            vocab_size = vocab_size, emb_size=emb_size, 
            hidden_size=hidden_size, output_size=output_size, dropout=dropout,
            dependent_z=dependent_z, layer=layer,
            selection=selection, lasso=lasso,
            lagrange_alpha=lagrange_alpha, lagrange_lr=lagrange_lr,
            lambda_init=lambda_init,
            lambda_min=lambda_min, lambda_max=lambda_max)

            initialize_model_(latent_model)

            # load pre-trained word embeddings
            with torch.no_grad():
                latent_model.embed.weight.data.copy_(torch.from_numpy(vectors))
                print("Embeddings fixed: {}".format(cfg["fix_emb"]))
                latent_model.embed.weight.requires_grad = not cfg["fix_emb"]

            latent_model = latent_model.to(device)
            self.latent_models.append(latent_model)

            
            #Try to use embedding
            aspect_rating_classifier = Rating_Classifier(
                hidden_size=hidden_size, output_size=output_size,
                dropout=dropout, layer=layer)
            aspect_rating_classifier = aspect_rating_classifier.to(device)    
            initialize_model_(aspect_rating_classifier)
            self.aspect_rating_classifiers.append(aspect_rating_classifier)

            aspect_polarity_classifier = Polarity_Classifier(
              hidden_size=hidden_size, output_size=output_size,
                dropout=dropout, layer=layer)
            aspect_polarity_classifier = aspect_polarity_classifier.to(device)    
            initialize_model_(aspect_polarity_classifier)
            self.aspect_polarity_classifiers.append(aspect_polarity_classifier)

            transformer = TransformerModel(embed=embed,hidden_size=hidden_size, nhead=2)
            transformer = transformer.to(device)
            initialize_model_(transformer)

            # # load pre-trained word embeddings
            # with torch.no_grad():
            #     transformer.embed.weight.data.copy_(torch.from_numpy(vectors))
            #     print("Embeddings fixed: {}".format(cfg["fix_emb"]))
            #     transformer.embed.weight.requires_grad = not cfg["fix_emb"]

            self.transformers.append(transformer)

        

    def forward(self, x):
        mask = (x != 1)  # [B,T]
        final_rating = torch.zeros(x.shape[0], 1, requires_grad=True)
        final_rating = final_rating.to(device)
        aspect_ratings = torch.zeros(x.shape[0], self.aspects)
        aspect_ratings = aspect_ratings.to(device)
        aspect_polarities = torch.zeros(x.shape[0], self.aspects)
        aspect_polarities = aspect_polarities.to(device)
        z_matrix = torch.zeros(x.shape[0],self.aspects, x.shape[1], dtype=torch.long)

        for i in range(self.aspects):
            latent_model = self.latent_models[i].latent_model
            z = latent_model(x, mask) 
            z = z.to(device)

            z_mask = z.squeeze(-1) > 0.5 # [B, T, 1] 
            
            transformer = self.transformers[0]
            transformer_output, y = transformer(x, mask, z)
            transformer_output = transformer_output.squeeze(0) #[B, words len]
            transformer_output = transformer_output.to(device)
    
            aspect_rating = self.aspect_rating_classifiers[0](transformer_output)
            aspect_ratings[:, i:i+1] = aspect_rating
            aspect_polarity = self.aspect_polarity_classifiers[0](y, transformer_output)
            aspect_polarities[:, i:i+1] = aspect_polarity

            z_mask_extened = z_mask.unsqueeze(1)
            z_matrix[:,i:i+1,:z.shape[1]] = z_mask_extened

        # normalize apsect polarity
        aspect_polarities_sum = torch.sum(aspect_polarities,1)
        aspect_polarities_sum = aspect_polarities_sum.unsqueeze(1)
        aspect_polarities = aspect_polarities / aspect_polarities_sum

        aspect_sum = torch.sum(aspect_ratings * aspect_polarities, 1)
        aspect_sum = aspect_sum.unsqueeze(1)

        print(aspect_ratings, aspect_polarities)
        final_rating = torch.add(final_rating, aspect_sum)

        return final_rating, z_matrix
        

    def get_loss(self, preds, targets, z_matrix, mask=None):
        loss = nn.MSELoss(reduction='none')
        loss = loss.to(device)
        preds = preds.to(device)
        targets = targets / 5.
        targets = targets.to(device)

        loss_mat = loss(preds, targets)
        loss_vec = loss_mat.mean(1)   # [B]
        classificaiton_loss = loss_vec.mean()   
        classificaiton_loss = classificaiton_loss.to(device)

        total_loss = classificaiton_loss #+ self.L2_regularize * z_matrix_loss
        total_loss = total_loss.to(device)

        z_loss = 0.
        
        for i in range(self.aspects):
            z_regularization, _ = self.latent_models[i].get_loss(preds, targets, mask=mask)
            z_loss =  z_loss + z_regularization.item()

        z_matrix = z_matrix.float() # 64 x 700 x 5 
        z_matrix_t = z_matrix.permute(0,2, 1)
        zzT = torch.matmul(z_matrix, z_matrix_t)
        zzT_diag = torch.triu(zzT, diagonal=1)
        z_matrix_loss = torch.sum(zzT_diag) * self.L2_regularize
        total_loss = total_loss + z_matrix_loss + z_loss
        
        
        return classificaiton_loss, total_loss, z_loss, z_matrix_loss