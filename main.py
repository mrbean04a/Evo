import torch
import json
import random
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import cached, TTLCache

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    KNOWLEDGE_BASE_PATH = 'knowledge_base.json'
    FACT_RETRIEVAL_THRESHOLD = 0.5
    POPULATION_SIZE = 10
    GENERATIONS = 5
    MUTATION_RATE = 0.1

class Brain:
    def __init__(self):
        self.character_name = "eve"
        print("Hello World")

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.model.to(Config.DEVICE)

        with open(Config.KNOWLEDGE_BASE_PATH, 'r') as file:
            self.knowledge_base = json.load(file)

        self.character_state = "child"
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.update_knowledge_vectors()

        self.cache = TTLCache(maxsize=100, ttl=3600)  
        self.user_profiles = {}  # Initialize user profiles storage

        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.memory = {}  # Session-based memory
        self.long_term_memory = {}  # Long-term memory

        self.self_evaluation()

    def set_character_name(self, name):
        self.character_name = name
        print(f"Character name set to: {self.character_name}")

    def self_evaluation(self):
        try:
            test_query = "What is AI?"
            test_result = self.search_knowledge_base(test_query)
            assert test_result is not None

            test_web_data = self.fetch_web_data(test_query)
            assert isinstance(test_web_data, str) and len(test_web_data) > 0

            initial_state = self.character_state
            self.evolve_character()
            assert self.character_state != initial_state

            self.apply_genetic_algorithm()

            print("All functions are working properly.")
        except Exception as e:
            print(f"Self-evaluation failed: {e}")

    @cached(cache=TTLCache(maxsize=100, ttl=3600))
    def fetch_web_data(self, query):
        """Fetch data from the web using Google Search."""
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([para.get_text() for para in paragraphs])

    def update_knowledge_base(self, new_data):
        self.knowledge_base.append({'text': new_data})
        with open(Config.KNOWLEDGE_BASE_PATH, 'w') as file:
            json.dump(self.knowledge_base, file)
        self.update_knowledge_vectors()

    def update_knowledge_vectors(self):
        texts = [entry['text'] for entry in self.knowledge_base]
        self.knowledge_vectors = self.vectorizer.fit_transform(texts)

    def simulate_human_learning(self, new_data):
        self.update_knowledge_base(new_data)

    def generate_response(self, input_text):
        inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(Config.DEVICE)
        with torch.no_grad():
            outputs = self.model(inputs)
        response = self.tokenizer.decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
        return response

    def handle_query(self, user_id, input_text):
        """Handle user queries and provide appropriate responses."""
        response = self.search_knowledge_base(input_text)
        if response:
            return response

        new_data = self.fetch_web_data(input_text)
        if new_data:
            self.simulate_human_learning(new_data)

        response = self.generate_response(input_text)
        self.memory[user_id] = input_text
        if user_id in self.long_term_memory:
            self.long_term_memory[user_id].append(input_text)
        else:
            self.long_term_memory[user_id] = [input_text]

        return response

    def search_knowledge_base(self, input_text):
        input_vector = self.vectorizer.transform([input_text])
        similarities = cosine_similarity(input_vector, self.knowledge_vectors)
        best_match_idx = np.argmax(similarities)
        if similarities[0, best_match_idx] > Config.FACT_RETRIEVAL_THRESHOLD:
            return self.knowledge_base[best_match_idx]['text']
        return None

    def evolve_character(self):
        evolution_stages = ['child', 'teen', 'adult']
        current_stage_index = evolution_stages.index(self.character_state)
        if current_stage_index < len(evolution_stages) - 1:
            self.character_state = evolution_stages[current_stage_index + 1]

    def apply_genetic_algorithm(self):
        population = [self.create_random_parameters() for _ in range(Config.POPULATION_SIZE)]
        for generation in range(Config.GENERATIONS):
            scores = [self.evaluate_fitness(params) for params in population]
            best_params = self.select_best_parameters(population, scores)
            self.reproduce(population, best_params)
        self.model_parameters = best_params

    def create_random_parameters(self):
        return {
            'learning_rate': random.uniform(1e-5, 1e-1),
            'batch_size': random.choice([8, 16, 32])
        }

    def evaluate_fitness(self, params):
        train_texts = [
            'Translate English to French: Hello, how are you?',
            'Translate English to French: I love programming.'
        ]
        train_labels = [
            'Bonjour, comment ça va?',
            'J\'aime programmer.'
        ]

        val_texts = [
            'Translate English to French: Good morning!',
            'Translate English to French: What is your name?'
        ]
        val_labels = [
            'Bonjour!',
            'Comment vous appelez-vous?'
        ]

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
        train_labels = self.tokenizer(train_labels, truncation=True, padding=True, return_tensors='pt')['input_ids']

        train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=params['learning_rate'])
        self.model.train()
        for epoch in range(1):
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                optimizer.zero_grad()
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        val_labels = self.tokenizer(val_labels, truncation=True, padding=True, return_tensors='pt')['input_ids']
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(val_encodings['input_ids'].to(Config.DEVICE))
            accuracy = np.mean([torch.equal(output, label) for output, label in zip(outputs, val_labels)])
        return accuracy

    def select_best_parameters(self, population, scores):
        best_idx = np.argmax(scores)
        return population[best_idx]

    def reproduce(self, population, best_params):
        new_population = [self.mutate(best_params) for _ in range(len(population))]
        population[:] = new_population

    def mutate(self, params):
        mutation_rate = Config.MUTATION_RATE
        new_params = params.copy()
        if random.random() < mutation_rate:
            new_params['learning_rate'] += random.uniform(-0.01, 0.01)
            new_params['learning_rate'] = max(1e-5, min(new_params['learning_rate'], 1e-1))
        if random.random() < mutation_rate:
            new_params['batch_size'] = random.choice([8, 16, 32])
        return new_params

    def simulate_human_thinking(self, input_text):
        if self.character_state == 'child':
            response = f"As a young learner, here's what I found: {input_text}"
        elif self.character_state == 'teen':
            response = f"Considering my experience, here's the information on: {input_text}"
        else:
            response = f"Based on extensive knowledge and experience, here's a detailed response: {input_text}"
        return response

    def store_user_profile(self, user_id, preferences, interests):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {'preferences': {}, 'interests': [], 'feedback': []}
        self.user_profiles[user_id]['preferences'].update(preferences)
        self.user_profiles[user_id]['interests'].extend(interests)
        self.user_profiles[user_id]['interests'] = list(set(self.user_profiles[user_id]['interests']))

    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {})

    def collect_feedback(self, user_id, query, response, feedback):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {'preferences': {}, 'interests': [], 'feedback': []}
        self.user_profiles[user_id]['feedback'].append({
            'query': query,
            'response': response,
            'feedback': feedback
        })
        self.analyze_feedback(user_id)

    def analyze_feedback(self, user_id):
        feedbacks = self.user_profiles[user_id]['feedback']
        positive_feedbacks = [fb for fb in feedbacks if fb['feedback'] == 'positive']
        negative_feedbacks = [fb for fb in feedbacks if fb['feedback'] == 'negative']

        if len(positive_feedbacks) > len(negative_feedbacks):
            print(f"User {user_id} is generally satisfied with the responses.")
        else:
            print(f"User {user_id} is generally dissatisfied with the responses. Need to improve.")

    def perform_sentiment_analysis(self, text):
        result = self.sentiment_analyzer(text)
        return result[0]['label'], result[0]['score']

    def remember_context(self, user_id, context):
        self.memory[user_id] = context

    def recall_context(self, user_id):
        return self.memory.get(user_id, '')

    def remember_long_term(self, user_id, information):
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = []
        self.long_term_memory[user_id].append(information)

    def recall_long_term(self, user_id):
        return self.long_term_memory.get(user_id, [])

if __name__ == "__main__":
    brain = Brain()
    
    # Example usage
    brain.set_character_name("AI Assistant")
    
    user_id = "user123"
    
    # Store user profile
    brain.store_user_profile(user_id, preferences={'language': 'English'}, interests=['AI', 'technology'])
    
    # Handle a query
    query = "What is the capital of France?"
    response = brain.handle_query(user_id, query)
    print(response)
    
    # Collect feedback
    brain.collect_feedback(user_id, query, response, 'positive')
    
    # Perform sentiment analysis
    sentiment, score = brain.perform_sentiment_analysis("I love this AI assistant!")
    print(f"Sentiment: {sentiment}, Score: {score}")
    
    # Remember and recall context
    brain.remember_context(user_id, "Discussing geography")
    print(brain.recall_context(user_id))
    
    # Long-term memory
    brain.remember_long_term(user_id, "Paris is the capital of France.")
    print(brain.recall_long_term(user_id))