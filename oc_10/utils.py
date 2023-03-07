class UtteranceExtractor:
    """
    Extracts utterances from a given dialog data
    :param data_turns: A series containing dialog data turns
    :return: A list of utterances extracted from the dialog data
    """

    def __init__(self, data_turns):
        self.data_turns = data_turns

        self.dict_keys = {'or_city': [], 'dst_city': [], 'str_date': [], 'end_date': [], 'budget': []}

    def _setDictKeys(self):
        # Initialisation d'un dictionnaire pour stocker les clés comme
        # - or_city 
        # - dst_city
        # - str_date
        # - end_date
        # - budget
        # et leurs valeurs dans une liste à parir de tout les messages par utilisateur

        #self.dict_keys = {'or_city': [], 'dst_city': [], 'str_date': [], 'end_date': [], 'budget': []}
        

        # Pour chaque tour dans la liste de tours
        for turn in self.data_turns:
            # Pour chaque message dans le tour
            for message in turn:
                # Si le message n'est pas de l'utilisateur, passer au message suivant
                if message['author'] != 'user':
                    continue

                # Récupération des acts associés au message
                acts = message['labels']['acts']

                # Pour chaque act associé au message
                for act in acts:
                    # Pour chaque argument associé à l'act
                    for arg in act['args']:
                        # Récupération de la clé et de la valeur à partir de l'argument
                        key, value = self._getKeyValue(arg)

                        # Si la clé n'est pas parmi les clés recherchées, passer au argument suivant
                        if key not in self.dict_keys.keys():
                            continue

                        # Ajout de la valeur associée à la clé dans le dictionnaire
                        self.dict_keys[key].append(value)

    def _getKeyValue(self, arg):
        """
        Extracts the key and value from a given argument
        :param arg: A dictionary containing an argument
        :return: A tuple containing the key and value of the argument
        """
        key = arg.get("key")
        value = arg.get("val")
        if key and value and value != '-1':
            return (key, value)
        return (None, None)

    def _getEntityLabels(self, text, key, value):
        """
        Extracts the entity label for a given key-value pair in a given text
        :param text: The text containing the key-value pair
        :param key: The key for the entity label
        :param value: The value for the entity label
        :return: A dictionary containing the extracted entity label
        """
        text = text.lower()
        value = value.lower()

        return {
            'entity_name': key,
            'start_char_index': text.index(value),
            'end_char_index': text.index(value) + len(value)
        }
    
    def getEntitiesFromUtterances(self, id_utt):
        utt = self.utterances[id_utt]
        # Extract the text of the message
        text = utt['text']
        
        # Create a dictionary with the entity_name as key and the entity mention as value
        y_true = {entity['entity_name']: text[entity['start_char_index']:entity['end_char_index']]
                for entity in utt['entity_labels']}
        
        # Return the dictionary of entities
        return y_true

    def extract(self):
        self.utterances = []
        # Loop through each turn in the dialog data
        for turn in self.data_turns:
            for message in turn:
                # Only extract utterances from user messages
                if message['author'] != 'user':
                    continue

                text = message['text']
                entity_labels = []
                # Loop through each action in the message's labels
                for act in message['labels']['acts']:
                    for arg in act['args']:
                        key, value = self._getKeyValue(arg)

                        # Only extract entity labels for valid entity keys
                        if key not in self.dict_keys.keys() or key == None or value == None:
                            continue

                        entity_labels.append(self._getEntityLabels(text, key, value))

                # Append the utterance if at least one entity label was extracted
                if entity_labels:
                    self.utterances.append({
                        'text': text,
                        'intent_name': 'BookFlight',
                        'entity_labels': entity_labels
                    })

