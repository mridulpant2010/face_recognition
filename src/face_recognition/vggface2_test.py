import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 1. Initialize the Face Detector (MTCNN)
# VGGFace2 models expect images to be cropped to 160x160 pixels.
mtcnn = MTCNN(image_size=160, margin=0)

# 2. Load the model PRE-TRAINED ON VGGFace2
# This automatically downloads the weights trained on the ox-vgg/vgg_face2 dataset
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_vggface2_embedding(image_path):
    # Open the image
    img = Image.open(image_path).convert('RGB')
    
    # Detect the face and crop it
    # MTCNN returns a tensor of the cropped face ready for the neural network
    face_tensor = mtcnn(img)
    
    # TODO: let us visualize what all the face tensor contains.
    
    if face_tensor is None:
        print(f"No face detected in {image_path}")
        return None
        
    # Add a batch dimension (Shape becomes: [1, 3, 160, 160])
    face_tensor = face_tensor.unsqueeze(0)
    
    # Extract the 512-Dimensional Embedding
    with torch.no_grad():
        embedding = resnet(face_tensor)
        
    return embedding

# --- Run the Verification ---
img1_path = r'C:\coding\face_recognition\src\dataset\bikeride.jpg'
img2_path = r'C:\coding\face_recognition\src\dataset\friends.jpg'

emb1 = get_vggface2_embedding(img1_path)
emb2 = get_vggface2_embedding(img2_path)

if emb1 is not None and emb2 is not None:
    # Compare the two embeddings using Cosine Similarity
    similarity = F.cosine_similarity(emb1, emb2).item()
    
    print(f"VGGFace2 Cosine Similarity: {similarity:.4f}")
    
    # For VGGFace2 in this library, a good starting threshold is around 0.6
    if similarity > 0.60:
        print("Verification: SAME PERSON")
    else:
        print("Verification: DIFFERENT PEOPLE")