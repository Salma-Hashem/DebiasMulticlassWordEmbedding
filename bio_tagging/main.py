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
    terms = sample_embeddings.split();
    current_word = terms[0];
    embeddings_dict[current_word] = terms[1:];


    

basic_categories = ["token", "pos", "bio"]
all_lines_dict = dict();
#FEATURES_LIST = ["token", "pos", "bio", "is_capital", "is_punctuation"]
#when you add an item in this list, instances where the feature would be blank
#are instead replaced with START or END, depending on where the blank feature would occur in the sentence
#TODO: test if features_list works with empty list
FEATURES_LIST = []
template = ["token", "pos", "bio", "is_punctuation", "is_capital"]
#if track_ending  is disabled, no features will include the START or END labels (they'll just be excluded)
track_ending = True;

line_count = sum(1 for line in read_file);
#template_dict = {"token":"*NONE*", "pos":"*NONE*", "bio":"*NONE*"}
template_dict = dict.fromkeys(template, "*NONE*")
all_lines_dict = dict.fromkeys(range(line_count));
for key in all_lines_dict:
    all_lines_dict[key] = template_dict.copy();
current_line_dex = -1;
read_file.seek(0);
#thresholds for features and maximum ngram distance
max_line_tracking_distance = 3;
token_count = 0;
big_sentence_threshold = 15;
small_sentence_threshold = 4;
small_word_threshold = 3;
large_word_threshold = 10;
#boolean declarations for certain features
contains_extra_capital = "false"
contains_quote = "false";
contains_hyphenation = "false";
contains_big_word = "false";
contains_small_word = "false";
contains_comma = "false";
for line in read_file:
    current_line_dex +=1;
    current_line_dict = dict();
    parts_of_line = line.split("\t");
    line_length = len(parts_of_line);
    if not line.isspace():
        for line_dex in range(line_length):
            current_category = basic_categories[line_dex];
            current_line_part = parts_of_line[line_dex];
            current_line_dict[current_category] = current_line_part.strip();
    if "token" in current_line_dict:
        #checking for word-conditions
        #that affect the whole phrase
        if token_count != 1 and current_line_dict["token"][0].isupper():
            contains_extra_capital = "true";
        stem = ps.stem(current_line_dict["token"])
        newlemma = lemma.lemmatize(current_line_dict["token"]);
        current_line_dict["lemma"] = newlemma;
        current_line_dict["stem"] = stem;
        synonyms = [];
        antonyms = [];
        syncount = 0;
        antcount = 0;
        maxsyn = 4;
        maxant = 4;
        for syn in wordnet.synsets(current_line_dict["token"]):
            for lem in syn.lemmas():
                #current_line_dict["synonym_"+str(syncount)] = lem.name();
                synonyms.append(lem.name());
                syncount+=1;
                if syncount > maxsyn:
                    break;
                if lem.antonyms():
                    #current_line_dict["antonym_" + str(antcount)] = lem.antonyms()[0].name();
                    antonyms.append(lem.antonyms()[0].name());
                    antcount+=1;
                    if antcount > maxant:
                        break;
        current_line_dict["synonym"]= synonyms;
        current_line_dict["antonym"] = antonyms;
        if current_line_dict["token"] == "''" or current_line_dict["token"] == "``":
            contains_quote = "true";
        if current_line_dict["token"] == ",":
            contains_comma = "true";
        if len(current_line_dict["token"]) >= large_word_threshold:
            contains_big_word = "true";
        if len(current_line_dict["token"]) <= small_word_threshold:
            contains_small_word = "true";
        if "-" in current_line_dict["token"]:
            contains_hyphenation = "true";

    if line.isspace():
        #print("newline found")
        is_question = "false"
        is_exclamation = "false"
        #checking for conditions that relate
        #to the endings of a phrase:
        if current_line_dex-1 in all_lines_dict:
            if all_lines_dict[current_line_dex -1]["token"] == "?":
                is_question = "true"
            if all_lines_dict[current_line_dex -1]["token"] == "!":
                is_exclamation = "true"
        for index in range(token_count+1):
            all_lines_dict[current_line_dex-index]["is_question"] =is_question;
            all_lines_dict[current_line_dex-index]["is_exclamation"] =is_exclamation;
            all_lines_dict[current_line_dex-index]["contains_extra_capital"] = contains_extra_capital;
            all_lines_dict[current_line_dex-index]["contains_quote"] = contains_quote;
            all_lines_dict[current_line_dex-index]["contains_comma"] = contains_comma;
            all_lines_dict[current_line_dex-index]["contains_big_word"] = contains_big_word;
            all_lines_dict[current_line_dex-index]["contains_small_word"] = contains_small_word;
            all_lines_dict[current_line_dex-index]["contains_hyphenation"] = contains_hyphenation;

        if token_count >= big_sentence_threshold:
            current_line_dict["big_sentence"] = "true";
            temp_dex = current_line_dex;
            for index in range(token_count+1):
                all_lines_dict[current_line_dex-index]["big_sentence"] = "true";
        if token_count < big_sentence_threshold:
            current_line_dict["big_sentence"] = "false";
            temp_dex = current_line_dex;
            for index in range(token_count):
                all_lines_dict[current_line_dex-index]["big_sentence"] = "false";
        if token_count <= small_sentence_threshold:
            current_line_dict["small_sentence"] = "true";
            temp_dex = current_line_dex;
            for index in range(token_count+1):
                all_lines_dict[current_line_dex-index]["small_sentence"] = "true";
        if token_count > small_sentence_threshold:
            current_line_dict["small_sentence"] = "false";
            temp_dex = current_line_dex;
            for index in range(token_count):
                all_lines_dict[current_line_dex-index]["small_sentence"] = "false";
        token_count = 0;
        #reset the boolean values for various
        #feature trackers
        contains_extra_capital = "false";
        contains_quote = "false"
        contains_comma = "false"
        contains_big_word = "false";
        contains_hyphenation = "false";
        contains_small_word = "false";

        for surrounding_dex in range(current_line_dex - max_line_tracking_distance, current_line_dex+ max_line_tracking_distance + 1):
            if surrounding_dex in all_lines_dict:
                surrounding_line_dict = all_lines_dict[surrounding_dex];
                #print(surrounding_line_dict);
                if surrounding_dex < current_line_dex:
                    surrounding_line_dict["right_bound"] = current_line_dex-1;
                if surrounding_dex > current_line_dex:
                    surrounding_line_dict["left_bound"] = current_line_dex +1;
    else:
        token_count +=1;
        if current_line_dict["token"][0].isupper():
            current_line_dict["is_capital"]="true";
        else:
            current_line_dict["is_capital"]="false";
        if current_line_dict["token"][0] in string.punctuation:
            current_line_dict["is_punctuation"] = "true";
        else:
            current_line_dict["is_punctuation"] = "false";
        if current_line_dict["token"] == ",":
            current_line_dict["is_comma"] = "true";
        else:
            current_line_dict["is_comma"] = "false";
        if len(current_line_dict["token"]) <= small_word_threshold:
            current_line_dict["is_small_word"] = "true";
        else:
            current_line_dict["is_small_word"] = "false";
        if "-" in current_line_dict["token"]:
            current_line_dict["is_hypenated"] = "true";
        else:
            current_line_dict["is_hyphenated"] = "false";
        if len(current_line_dict["token"]) >= large_word_threshold:
            current_line_dict["is_large_word"] = "true";
        else:
            current_line_dict["is_large_word"] = "false";
        if "left_bound" not in all_lines_dict[current_line_dex]:
            all_lines_dict[current_line_dex]["left_bound"] = max((current_line_dex - max_line_tracking_distance),0);
        if "right_bound" not in all_lines_dict[current_line_dex]:
            all_lines_dict[current_line_dex]["right_bound"] = min((current_line_dex + max_line_tracking_distance),line_count-1);
            
    if line_length == 0:
        #something is probably wrong if this happens
        current_line_dict["token"] = "line_tokens_not_found"
    all_lines_dict[current_line_dex].update(current_line_dict);

exclusion_list = ["left_bound", "right_bound", "is_comma"];
individual_features = ["is_capital", "token", "is_punctuation", "is_comma", "is_hyphenated", "is_small_word", "is_large_word", "stem", "lemma"]
exclude_current = ["bobjones"]
exclude_current = ["bio", "token"]
#exclude_not_prev1 = ["bio","is_capital", "is_punctuation", "token"];
exclude_not_prev1 = ["bobjones"];
#exclude_beyond_dist_one = ["is_capital", "is_punctuation"]
#exclude_not_prev1 = individual_features;
exclude_beyond_dist_one = ["bobjones"]
#exclude_beyond_dist_one = individual_features;
#exclude_not_current = ["small_sentence", "stem", "big_sentence", "is_question", "is_exclamation", "contains_extra_capital", "contains_quote", "contains_hyphenation", "contains_big_word"]
exclude_not_current = ["small_sentence", "big_sentence", "is_question", "is_exclamation", "contains_hyphenation", "contains_big_word", "contains_small_word", "contains_extra_capital", "contains_quote", "contains_comma"]
exclusion_list += exclude_not_current;
def generate_feature_line(master_dictionary, line_index, lower_bound, upper_bound):
    output_line = "";
    for surrounding_dex in range(line_index - max_line_tracking_distance, line_index + max_line_tracking_distance+1):
        if surrounding_dex < line_index:
            prefix = "prev_" + str(abs(line_index-surrounding_dex));
        if surrounding_dex == line_index:
            prefix = "current_"
        if surrounding_dex > line_index:
            prefix = "next_" + str(abs(line_index-surrounding_dex));
        if lower_bound <= surrounding_dex <= upper_bound:
            current_dict = master_dictionary[surrounding_dex];
            for key in current_dict:
                #skip conditions:
                if key in exclusion_list:
                    continue;
                if surrounding_dex == line_index and key in exclude_current:
                    continue;
                if not (key == "synonym" or key == "antonym"):
                    if current_dict[key].isspace():
                        continue;
                if (not (surrounding_dex == line_index -1 )) and key in exclude_not_prev1:
                    continue;
                if (abs(surrounding_dex - line_index) > 1) and key in exclude_beyond_dist_one:
                    continue;
                if (not (surrounding_dex == line_index)) and key in exclude_not_current:
                    continue;
                #experimenting with removing numbers from synonyms/antonyms feature category
                if key == "synonym" or key == "antonym":
                    for ant_or_syn in current_dict[key]:
                        current_feature = prefix + key + "=" + ant_or_syn + "\t";
                        output_line += current_feature;
                else:
                    current_val = current_dict[key];
                    current_feature = prefix + key + "="+ current_val.strip() + "\t"
                    output_line += current_feature;
        if track_ending == True:
            if surrounding_dex < lower_bound:
                for item in FEATURES_LIST:
                    current_feature = prefix + item + "="+ "BEGIN" + "\t"
                    output_line += current_feature;
            if surrounding_dex > upper_bound:
                for item in FEATURES_LIST:
                    current_feature = prefix + item + "="+ "END" + "\t"
                    output_line += current_feature;

    return output_line;
for line_dex in range(line_count):
    current_line_dict = all_lines_dict[line_dex];
    #current_token = current_line_dict["token"];
    #if not current_token == "\n" and not current_token.isspace():
    if "token" in current_line_dict:
        if current_line_dict["token"] == "*NONE*":
            write_file.write("\n");
            continue;
        #print("token found")
        current_token = current_line_dict["token"]
        feature_line = generate_feature_line(all_lines_dict, line_dex, current_line_dict["left_bound"], current_line_dict["right_bound"]);
        if "bio" in current_line_dict:
            final_line = current_line_dict["token"].strip() +  "\t" + feature_line + current_line_dict["bio"] + "\n"
            write_file.write(final_line)
        else:
            write_file.write("\n")
    else:
        write_file.write("\n");




