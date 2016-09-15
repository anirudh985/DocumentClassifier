# DocumentClassifier

1. Document parsing using beautiful soup
2. Stop word removal
3. Tokenizing and constructing corpus map
    Corpus map - >  (word, position),
    position is the column number that corresponds to the word in feature vector
4. Constructing feature vector of document
    Feature vector -> map of (word,count)
5. Storing the feature vector, (word -> position), (documentNumber->position in featureVectorFile ) in a file.
6. Retrieving the feature vector from file -> not required for current
7. Need to write a full feature vector file from our output files
