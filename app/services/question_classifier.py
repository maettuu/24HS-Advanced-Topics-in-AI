import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from enum import Enum
import os


class QuestionCategory(Enum):
    KNOWLEDGE = 0
    MULTIMEDIA = 1
    RECOMMENDATION = 2
    SMALLTALK = 3


class ClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QuestionClassifier:
    def __init__(self) -> None:
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_mapping = {category.name.lower(): category.value for category in QuestionCategory}
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize and load the neural network
        model_path = os.path.join(os.path.dirname(__file__),
                                  './../../utils/test_scripts/question_classifier/question_classifier_model.pth')
        self.classifier = ClassifierNN(input_dim=384, num_classes=len(self.label_mapping))
        self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        print("Model loaded successfully!")

    def classify(self, question: str) -> QuestionCategory:
        # Embed the question
        embedding = torch.tensor(self.model.encode([question]), dtype=torch.float32).to(self.device)

        # Predict the category
        with torch.no_grad():
            output = self.classifier(embedding)
            prediction = torch.argmax(output, dim=1).item()

        # Map prediction to label
        return QuestionCategory(prediction)
