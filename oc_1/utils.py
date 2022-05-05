def getKeysByValue(dictOfElements, valueToFind):
    '''
    Get a list of keys from dictionary which has the given value
    Input:
        dictOfElements - dictionary
        valueToFind - value to find

    Output:
        listOfKeys - list of keys containing the selected value
    '''
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def get_linenr_text(filename):
    '''
    Parameters
    ----------
    filename : string
        directory to the text file.

    Returns
    -------
    dict_text : dictionary
        The dictionary : with keys - the number of line and value - the text from one line.

    '''
    dict_text = {}
    with open(filename) as file:
        i=0
        for line in file:
            dict_text[i] = line.rstrip()
            i+=1
    return dict_text
