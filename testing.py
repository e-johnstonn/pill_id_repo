from deeplake.core.vectorstore.deeplake_vectorstore import DeepLakeVectorStore
import torch
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(pretrained=True)


return_nodes = {'avgpool': 'embedding'}

model = create_feature_extractor(model, return_nodes=return_nodes)

model.eval()
model.to(device)

tform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], dim=0) if x.shape[0] == 1 else x),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def embedding_function(images, model=model, transform=tform, batch_size=4):
    embeddings = []

    if isinstance(images, str):
        images = [images]

    for i in range(0, len(images), batch_size):
        batch = torch.stack([transform(Image.open(img)) for img in images[i:i+batch_size]])
        batch = batch.to(device)
        with torch.no_grad():
            embeddings += model(batch)['embedding'][:,:,0,0].cpu().numpy().tolist()

    return embeddings





if __name__ == '__main__':
        vector_store_path = ''
        vector_store = DeepLakeVectorStore(path=vector_store_path, read_only=True)
        image_path = ''
        result = vector_store.search(embedding_data=image_path, embedding_function=embedding_function, return_view=True)

        print(result.save_view())











