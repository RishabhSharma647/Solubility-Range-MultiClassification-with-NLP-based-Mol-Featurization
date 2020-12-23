""" Train MLP Solubility Range Classifier """
import mol2vec
import numpy as np
import pandas as pd
import rdkit
import tensorflow as tf

from gensim.models import word2vec
from MLP_Solubility_Range_Classifier import MLP_Classifier 
from mol2vec.features import DfVec
from mol2vec.features import mol2alt_sentence
from mol2vec.features import mol2sentence
from mol2vec.features import MolSentence
from mol2vec.features import sentences2vec
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras


# Import processed solubility data file (containing SMILES valid for Mol2vec vectorization) 
# (Subset of AqSolDB)
sol_data = pd.read_csv('processed_sol_data.csv')

# Classify solubility values into range labels
all_sols = list(sol_data['Solubility_logS'])

# Define function to map solubility values to range classes
# (Defined in order to enforce fair class balance along with range values)
def map_solubility_to_range(sol):
    if min(all_sols) <= sol <= -5:
        return 0
    elif -5 < sol <= -3:
        return 1
    elif -3 < sol <= -1.3:
        return 2
    elif -1.3 < sol <= max(all_sols):
        return 3
    
# Sort solubility values into solubility range labels
range_labels = [map_solubility_to_range(sol) for sol in all_sols]
# Add column for range labels to dataframe
sol_data['range_labels'] = range_labels

# Add one hot encodings 
sol_data['one_hot_1'] = [1.0 if sol_data['range_labels'][i] == 0 else 0.0 for i in range(sol_data.shape[0])]
sol_data['one_hot_2'] = [1.0 if sol_data['range_labels'][i] == 1 else 0.0 for i in range(sol_data.shape[0])]
sol_data['one_hot_3'] = [1.0 if sol_data['range_labels'][i] == 2 else 0.0 for i in range(sol_data.shape[0])]
sol_data['one_hot_4'] = [1.0 if sol_data['range_labels'][i] == 3 else 0.0 for i in range(sol_data.shape[0])]

# Import pre-trained mol2vec model
mol2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')

# Define function to convert SMILES strings to 300d embeddings
def smiles2vector_duplicates_average(smiles_string):
    """ Convert SMILES to 300d embedding
    
    Args:
        smiles_string (string): single SMILES string
    
    Returns:
        embedding (numpy.ndarray): 300d mol vector array
        
    """
    sentence = mol2alt_sentence(Chem.MolFromSmiles(smiles_string), radius = 1)
    vec_node = 0
    for i in range(len(sentence)):
        vec = mol2vec_model.wv[sentence[i]]
        vec_node += vec

    return vec_node / len(sentence)


# Create traning data
all_rows_df = [list(smiles2vector_duplicates_average(smile)) for smile in sol_data['SMILES']]
columns = ['mol2vec_' + str(i) for i in range(1, 301)]
df_inputs = pd.DataFrame(all_rows_df, columns = columns)

# Define input/output array 
X = df_inputs.iloc[:, 0:300].values
y = sol_data.iloc[:, 4:8].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_trans = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# Define hyper parameters
hidden_layers = {'h1': [200, 'tanh'], 'h2': [100, 'tanh'], 
                 'h3': [50, 'tanh'], 'output': [4, 'softmax']}
dropout = 0.3
loss = 'categorical_crossentropy'
optimizer = 'adam'
metric = 'accuracy'
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
batch_size = 100
num_epochs = 200
validation_split = .2

# Instantiate and create model
mlp_sol_classifier = MLP_Classifier(hidden_layers, dropout)

# Compile model
mlp_sol_classifier.compile_model(loss, optimizer, metric)

# Fit model
mlp_sol_classifier.fit_model(callbacks_list, batch_size, num_epochs, validation_split, X_train_trans, y_train)


# load model with best validation accuracy and test
model_best = keras.models.load_model("weights-improvement-200-0.73.hdf5")
results = model_best.evaluate(X_test, y_test)  
print(f"Test Loss: {results[0]:.3f}, Test Accuracy: {results[1]:.3f}")



    