import string;
import sys;
import nltk;
#ps = nltk.stem.PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()
ps = nltk.stem.SnowballStemmer("english")
from nltk.corpus import wordnet

read_file_name = sys.argv[1];
write_file_name = sys.argv[2];

read_file = open(read_file_name, "r");
write_file = open(write_file_name, "w");

sample_embeddings = open("sample_embeddings.embed", "r");
embeddings_dict = dict();

for line in sample_embeddings:
    terms = line.split();
    current_word = terms[0];
    embeddings_dict[current_word] = terms[1:];
read_file_dict = dict();
features_dict = dict();
line_index = 0;
for current_line in (read_file):
    #current_line = line;
    read_file_dict[line_index] = dict();
    features_dict[line_index] = dict();
    if current_line.isspace():
        read_file_dict[line_index]["token"] = "\n";
    else:
        line_parts = current_line.split();
        #initialize a dict of terms for the current line,
        #and a dict of features for the current line
        features_dict[line_index] = dict();
        for i in range(len(line_parts)):
            match i:
                case 0:
                    read_file_dict[line_index]["token"] = line_parts[i];
                case 1:
                    read_file_dict[line_index]["pos"] = line_parts[i];
                case 2:
                    read_file_dict[line_index]["bio_tag"] = line_parts[i];
        lowercase_word = line_parts[0].lower();
        #if line_parts[0] in embeddings_dict:
        if lowercase_word in embeddings_dict:
            #if the word is in our list of embeddings, add it to features
            #for index in range(len(embeddings_dict[line_parts[0]])):
            for index in range(len(embeddings_dict[lowercase_word])):
                #embedding_val = float(embeddings_dict[line_parts[0]][index]);
                embedding_val = float(embeddings_dict[lowercase_word][index]);

                if round(embedding_val) < 0:
                    embedding_val = -1;
                elif round(embedding_val) == 0:
                    embedding_val = 0;
                else:
                    embedding_val = 1;
                features_dict[line_index]["dim_" + str(index)] = str(embedding_val);
        features_dict[line_index]["pos"] = line_parts[1];
    line_index +=1;
trigram = False;
#bigram = False;
bigram = True;
if trigram:
    print("trigram activate!")
    #TODO: experiment with implementing
    #trigram for things like the pos tag
else:
    for line_index in range(len(read_file_dict)):
        line_dict = read_file_dict[line_index];
        if line_dict["token"].isspace():
            write_file.write(line_dict["token"]);
        else:
            current_features = features_dict[line_index];
            write_file.write(line_dict["token"] + "\t");
            for key, value in current_features.items():
                #todo: delete this line later
                write_file.write(str(key) + "=" + str(value) + "\t");
            #prev bio seems to make performance worse lol
            #consider commenting out prev_bio
            write_file.write("prev_bio=@@\t");
            if bigram:
                for surrounding_dex in range(max(0, line_index-2), min(line_index+3, len(read_file_dict))):
                    if surrounding_dex == line_index:
                        continue;
                    elif (line_index-surrounding_dex) <0:
                        prefix = "next"+ str(abs(line_index-surrounding_dex));
                    else:
                        prefix = "prev" + str(abs(line_index-surrounding_dex));
                    surrounding_dict = features_dict[surrounding_dex];
                    for (key, value) in surrounding_dict.items():
                        #does including weights of prev tokens matter? answer:
                        #if "dim" not in key: 
                        write_file.write(prefix + str(key) + "=" + str(value) + "\t");
                                             

                     

                        
                    
            if "bio_tag" in line_dict:
                write_file.write(line_dict["bio_tag"] + "\n");
            else:
                write_file.write("\n");

        











