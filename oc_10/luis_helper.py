from azure.cognitiveservices.language.luis.authoring import LUISAuthoringClient
from azure.cognitiveservices.language.luis.authoring.models import ApplicationCreateObject
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from typing import List, Tuple
from msrest.authentication import CognitiveServicesCredentials

import json, time, uuid
from dotenv import load_dotenv
import os

entities = ['or_city', 'dst_city', 'str_date', 'end_date', 'budget']  # list of entities to create
entities_map = {'or_city': 'geographyV2', 'dst_city': 'geographyV2', 'str_date': 'datetimeV2', 'end_date': 'datetimeV2', 'budget': 'number'}

class LUISHelper:
    def __init__(self):
        # Load the environment variables
        load_dotenv()
        try:
            self._authoring_key = os.environ.get("LUIS_AUTHORING_KEY")
            self._authoring_endpoint = os.environ.get("LUIS_AUTHORING_ENDPOINT")
        except:
            raise Exception('Please check the .env file, because LUIS_AUTHORING_KEY or LUIS_AUTHORING_ENDPOINT does not exists')
            
        
        if ('LUIS_PREDICTION_KEY' in os.environ) & ('LUIS_PREDICTION_ENDPOINT' in os.environ):    
            self._prediction_key = os.environ.get("LUIS_PREDICTION_KEY")
            self._prediction_endpoint = os.environ.get("LUIS_PREDICTION_ENDPOINT")
        else:
            print('You will not be able to make a prediction untill the LUIS_PREDICTION_KEY and LUIS_PREDICTION_ENDPOINT are not known by the application')
        
        # Initialize the LUIS client
        self.client = LUISAuthoringClient(
            self._authoring_endpoint, CognitiveServicesCredentials(self._authoring_key)
        )

    def create_app(self):
        """Creates a new LUIS app"""
        if 'LUIS_APP_VERSION' in os.environ:
            self.app_version = os.environ.get("LUIS_APP_VERSION")
        else:
            self.app_version = 0.1
        
        if 'LUIS_APP_NAME' in os.environ:
            self.app_name = os.environ.get("LUIS_APP_NAME")
        else:
            self.app_name = 'Book Flight'
        self.app_name += ' ' + str(uuid.uuid4())
        
        if 'LUIS_APP_DESCR' in os.environ:
            self.app_description = os.environ.get("LUIS_APP_DESCR")
        else:
            self.app_description = 'This is a Luis application'
        
        if 'LUIS_CULTURE' in os.environ:
            self.culture = os.environ.get("LUIS_CULTURE")
        else:
            self.culture = 'en-us'
        
        appDefinition = ApplicationCreateObject(name=self.app_name, initial_version_id=self.app_version, culture=self.culture)

        # create app
        self.app_id = self.client.apps.add(appDefinition)
        
        print(f"App {self.app_name} created with ID {self.app_id}")
        
    def load_app(self, app_id):
        self.app_id = app_id
        
        luis_app = self.client.apps.get(app_id=self.app_id)

        self.app_version = luis_app.active_version

    def add_prebuilt_entities(self, prebuilt_entities: list):
        """Adds prebuilt entities to the LUIS app"""
        # Add prebuilt entities
        for entity in prebuilt_entities:
            prebuilt_entity_id = self.client.model.add_prebuilt(
                app_id=self.app_id,
                version_id=self.app_version,
                prebuilt_extractor_names=[entity],
            )
            print(
                f"Prebuilt entity {entity} added with ID {prebuilt_entity_id} to app {self.app_id}"
            )
            
    def add_custom_entity(self, entity_name: str, entity_description: str = ''):
        """Adds a custom entity to the LUIS app"""
        entity_id = self.client.model.add_entity(
            self.app_id, self.app_version, name=entity_name, description=entity_description
        )
        print(f"Custom entity {entity_name} added with ID {entity_id} to app {self.app_id}")
        return entity_id

    def add_entity_feature(self, entity_id, feature_model_name: str):
        # Add entity feature
        #entity_id = self.client.model.get_entity_id(self.app_id, self.app_version, entity_name)
        feature_id = self.client.features.add_entity_feature(
            app_id=self.app_id,
            version_id=self.app_version,
            entity_id=entity_id,
            feature_relation_create_object={
                "model_name": feature_model_name,
            },
        )
        print(f"{feature_model_name} feature created with ID {feature_id} in entity id {entity_id}")
    
    def add_entities_and_features(self):
        for entity in entities:
            entity_id = self.add_custom_entity(entity)
            self.add_entity_feature(entity_id, entities_map[entity])
        
    def add_intent(self, intent_name: str):
        intent_id = self.client.model.add_intent(self.app_id, self.app_version, name=intent_name)
        print(f"{intent_name} intent created with id {intent_id}")
        
    def add_utterace(self, utteraces_data):
        for i in range(0, len(utteraces_data), 100):
            j = i + 100
            if j > len(utteraces_data):
                j = len(utteraces_data)

            self.client.examples.batch(
                        self.app_id,
                        self.app_version,
                        utteraces_data[i:j]
                    )
            
    def train_app(self):
        # Train the model
        print("Start training the app...")

        self.client.train.train_version(self.app_id, self.app_version)
        waiting = True

        while waiting:
            info = self.client.train.get_status(self.app_id, self.app_version)

            # get_status returns a list of training statuses, one for each model. Loop through them and make sure all are done.
            waiting = any(map(lambda x: 'Queued' == x.details.status or 'InProgress' == x.details.status, info))
            if waiting:
                print ("Waiting 10 seconds for training to complete...")
                time.sleep(10)
            else: 
                print("The app is trained !")
                waiting = False
                
    def publish_app(self):
        # Publish the app
        print("Start publishing the app...")

        self.client.apps.update_settings(self.app_id, is_public=True)
        publish_result = self.client.apps.publish(self.app_id, self.app_version, is_staging=False)

        print("The app is published.")

    def get_clientRuntime(self):
        runtimeCredentials = CognitiveServicesCredentials(self._prediction_key)
        clientRuntime = LUISRuntimeClient(endpoint=self._prediction_endpoint, credentials=runtimeCredentials)
        return clientRuntime

    def test(self, query: str):
        clientRuntime = self.get_clientRuntime()

        request = { "query" : query }

        predictionResponse = clientRuntime.prediction.get_slot_prediction(app_id=self.app_id, slot_name="Production", prediction_request=request)
        
        print("Top intent: {}".format(predictionResponse.prediction.top_intent))
        print("Intents: ")

        for intent in predictionResponse.prediction.intents:
            print("\t{}".format (json.dumps(intent)))
        print("Entities: {}".format (predictionResponse.prediction.entities))

    def get_predicted_entities(self, query: str):
        clientRuntime = self.get_clientRuntime()

        request = { "query" : query }
        response = clientRuntime.prediction.get_slot_prediction(app_id=self.app_id, slot_name="Production", prediction_request=request)

        entities = response.prediction.entities
        entities = {k:v[0] for k, v in entities.items()}

        return entities


class LUISEvaluation:

    def __init__(self, client: LUISRuntimeClient, app_id: str):
        # Load the environment variables
        self.client = client
        self.app_id = app_id
        self.y_pred = {}
        self.y_true = {}

    def _get_entities_utterances(self, data: dict) -> dict:
        """Helper function to extract entities from LUIS response"""
        text = data['text']
        y_true = {entity['entity_name']: text[entity['start_char_index']:entity['end_char_index']]
            for entity in data['entity_labels']}
        return y_true

    def _accuracy_score(self, y_pred: dict, y_true: dict) -> float:
        """Helper function to calculate accuracy score"""
        #correct = sum(y_pred.get(key) == y_true[key] for key in y_true.keys())


        def calculate_correct(y_pred, y_true):
            correct = 0
            for key in y_pred.keys():
                _pred = str(y_pred.get(key, ""))
                _true = str(y_true.get(key, ""))
                if _pred in _true:
                    correct += 0.5
                if _pred == _true:
                    correct += 0.5
            return correct

        correct = calculate_correct(y_pred, y_true)

        return (correct / len(y_true)) * 100

    def evaluate(self, test_data: List[dict]) -> Tuple[List[float], float]:
        """Evaluates the accuracy of a LUIS model on a list of test data"""
        scores = []
        for index, data in enumerate(test_data):
            clientRuntime = self.client.get_clientRuntime()

            request = { "query" : data['text'] }
            
            response = clientRuntime.prediction.get_slot_prediction(app_id=self.app_id, slot_name="Production", prediction_request=request)


            y_pred = {entity_key:entity_val[0] for entity_key, entity_val in response.prediction.entities.items()}

            y_true = self._get_entities_utterances(data)

            self.y_pred[index] = y_pred
            self.y_true[index] = y_true

            scores.append(self._accuracy_score(y_pred, y_true))
        mean_score = sum(scores) / len(scores)
        return scores, mean_score