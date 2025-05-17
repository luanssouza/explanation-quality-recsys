import argparse
import os
from easydict import EasyDict as edict
from models.PGPR.utils import get_tail_entity_name, LASTFM_RELATION_NAME
from myutils import *

class ML100KDatasetMapper(object):
    def __init__(self, args):
        self.args = args
        self.generate_dbpid_mlpid_mapping()
        self.generate_kg_entities()
        self.generate_kg_relations()
        self.generate_user_attributes_mappings()
        self.generate_train_test_split()

    def generate_dbpid_mlpid_mapping(self):
        dataset_name = self.args.dataset
        file = open(DATASET_DIR[dataset_name] + "/i2kg_map.txt", "r")
        dburl_to_mlid = {}
        reader = csv.reader(file, delimiter="\t")
        header = next(reader)
        for row in reader:
            mlid = int(row[1])
            name = row[2]
            dburl = row[3]
            dburl_to_mlid[dburl] = [mlid, name]
        file.close()

        file = open(DATASET_DIR[dataset_name] + "/e_map.txt", "r", encoding='latin-1')
        fileo = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "w+")
        writer = csv.writer(fileo, delimiter="\t")
        header = ["mlid", "dbid", "name", "dburl"]
        writer.writerow(header)
        reader = csv.reader(file, delimiter="\t")
        header = next(reader)
        for row in reader:
            dbid = int(row[0])
            dburl = row[2]
            if dburl not in dburl_to_mlid: continue
            mlid = dburl_to_mlid[dburl][0]
            name = dburl_to_mlid[dburl][1]
            writer.writerow([mlid, dbid, name, dburl])
        file.close()
        fileo.close()

    def generate_train_test_split(self):
        dataset_name = self.args.dataset

        train_uid_review_tuples = {}
        print("Loading train reviews...")
        with open(DATASET_DIR[dataset_name] + "/train.csv", 'r', encoding='latin-1') as reviews_file:
            reader = csv.reader(reviews_file, delimiter=',')
            next(reader, None)
            for row in reader:
                uid = int(row[0])
                if uid not in train_uid_review_tuples:
                    train_uid_review_tuples[uid] = []
                train_uid_review_tuples[uid].append((row[0], row[1], row[2], row[3]))
        for uid, reviews in train_uid_review_tuples.items():
            reviews.sort(key=lambda x: int(x[-1]))  # sorting from recent to older

        print("Writing train...")
        with open(DATASET_DIR[dataset_name] + "/train.txt", 'w+') as file:
            for _, user_reviews in train_uid_review_tuples.items():
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()

        test_uid_review_tuples = {}
        print("Loading test reviews...")
        with open(DATASET_DIR[dataset_name] + "/test.csv", 'r', encoding='latin-1') as reviews_file:
            reader = csv.reader(reviews_file, delimiter=',')
            next(reader, None)
            for row in reader:
                uid = int(row[0])
                if uid not in test_uid_review_tuples:
                    test_uid_review_tuples[uid] = []
                test_uid_review_tuples[uid].append((row[0], row[1], row[2], row[3]))
        for uid, reviews in test_uid_review_tuples.items():
            reviews.sort(key=lambda x: int(x[-1]))  # sorting from recent to older

        print("Writing test...")
        with open(DATASET_DIR[dataset_name] + "/test.txt", 'w+') as file:
            for _, user_reviews in test_uid_review_tuples.items():
                for review in user_reviews:
                    s = ' '.join(review)
                    file.writelines(s)
                    file.write("\n")
        file.close()
        
        print("Zipping train and test...")
        zip_file(DATASET_DIR[dataset_name] + "/train.txt")
        zip_file(DATASET_DIR[dataset_name] + "/test.txt")
        print("Loading reviews.. DONE")

    #Generate mappings from uid to sensible attributes for gender, age and occupation
    def generate_user_attributes_mappings(self):
        dataset_name = self.args.dataset
        users_id = []
        genders = []
        ages = []
        occupations = []
        with open(DATASET_DIR[dataset_name] + "/u.user", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                attributes = row[0].strip().split('|')
                users_id.append(attributes[0])
                genders.append(attributes[2])
                ages.append(attributes[1])
                occupations.append(attributes[3])
        file.close()

        #Write user_gender mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2gender.txt", 'w+') as file:
            for user, gender in zip(users_id, genders):
                file.write(user + "\t" + gender + "\n")
        file.close()

        # Write user_occupation mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2occupation.txt", 'w+') as file:
            for user, occupation in zip(users_id, occupations):
                file.write(user + "\t" + occupation + "\n")
        file.close()

        # Write user_age mapping
        with open(DATASET_DIR[dataset_name] + "/mappings/uid2age_map.txt", 'w+') as file:
            for user, age in zip(users_id, ages):
                file.write(user + "\t" + age + "\n")
        file.close()

    def generate_kg_entities(self):
        dataset_name = self.args.dataset
        #Creates a dict of sets to store all the extracted entitities for every differnt type
        kg_entities = edict(
            user=[set(), 'user.txt'],
            movie=[set(), 'movie.txt'],
            producer=[set(), 'producer.txt'],
            distributor=[set(), 'distributor.txt'],
            writer=(set(), 'writer.txt'),
            cinematographer=[set(), 'cinematographer.txt'],
            category=[set(), 'category.txt'],
            actor=[set(), 'actor.txt'],
            director=[set(), 'director.txt']
        )
        entity_path = DATASET_DIR[dataset_name] + "/entities/"
        if not os.path.isdir(entity_path):
            os.makedirs(entity_path)

        file = open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", "r")
        reader = csv.reader(file, delimiter='\n')
        db_pid2ml_pid = {}
        ml_pid2db_pid = {}
        ml_pid2metada = {}
        next(reader, None)
        for i, row in enumerate(reader):
            row = row[0].strip().split("\t")
            db_pid2ml_pid[int(row[1])] = int(row[0])
            ml_pid2db_pid[int(row[0])] = int(row[1])
            ml_pid2metada[int(row[0])] = [row[2], row[3]]
        file.close()
        kg_entities['movie'][0] = set(ml_pid2db_pid.keys())

        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + "/kg_final.txt", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            next(csv_reader, None)
            for row in csv_reader:
                head = int(row[0])
                tail = row[2]
                relation = int(row[1])
                if head not in db_pid2ml_pid: continue

                #movie_id = db_pid2ml_pid[head]
                tail_name = get_tail_entity_name(dataset_name, relation) #Retriving what is the tail of that relation

                #kg_entities['movie'][0].add(movie_id)
                kg_entities[tail_name][0].add(tail)
        file.close()

        # Write user entity
        with open(DATASET_DIR[dataset_name] + "/u.user", 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            for row in csv_reader:
                row = row[0].strip().split('|')
                uid = int(row[0])
                kg_entities.user[0].add(uid)

        new_id2old_id = {}
        with open(entity_path + "/user.txt", 'w+') as file:
            for idx, u in enumerate(kg_entities.user[0]):
                new_id2old_id[idx] = int(u)
                file.writelines(str(idx))
                file.write("\n")
        file.close()

        zip_file(entity_path + "/user.txt")

        with open(DATASET_DIR[dataset_name] + "/mappings/user_mappings.txt", 'w+') as file:
            header = ["kg_id", "ml1m_id"]
            file.write('\t'.join(header) + "\n")
            for new_id, old_id in new_id2old_id.items():
                file.write(str(new_id) + '\t' + str(old_id) + "\n")
        file.close()

        #Populate movie entity file (Done by itself due to is different structure)
        new_id2old_id = {}
        with open(entity_path + "/movie.txt", 'w+') as file:
            for idx, movie in enumerate(kg_entities['movie'][0]):
                new_id2old_id[idx] = int(movie)
                file.write(str(idx) + "\n")
        file.close()

        # newId (0...n), oldId(movilandID), entityId(jointkgentityid), entityNameDBPEDIA
        with open(DATASET_DIR[dataset_name] + "/mappings/product_mappings.txt", 'w+') as file:
            header = ["kg_id", "ml1m_id", "db_id", "name", "dbpedia_url"]
            file.write('\t'.join(header) + "\n")
            for new_id, old_id in new_id2old_id.items():
                entity_id = ml_pid2db_pid[old_id]
                file.write("\t".join([str(new_id), str(old_id), str(entity_id), ml_pid2metada[old_id][0], ml_pid2metada[old_id][1] + "\n"]))
        file.close()

        zip_file(entity_path + "/movie.txt")

        #Retrive the dblink associated to the entity id in the kg completion
        entity_id2dblink = {}
        entity_file = open(DATASET_DIR[dataset_name] + "/e_map.txt", "r", encoding='latin-1')
        reader = csv.reader(entity_file, delimiter="\t")
        next(reader, None)
        for row in reader:
            eid = int(row[0])
            dblink = row[2]
            entity_id2dblink[eid] = dblink

        #Populating other entities
        for entity_name in get_entities_without_user(dataset_name):
            if entity_name == 'movie': continue
            new_id2old_id = {}
            filename = entity_path + entity_name + '.txt'
            #Populate entities
            with open(filename, 'w+') as file:
                for idx, entity in enumerate(kg_entities[entity_name][0]):
                    new_id2old_id[idx] = int(entity)
                    file.write(str(idx) + "\n")
            file.close()

            # newId (0...n), entityId(jointkgentityid), entityNameDBPEDIA
            with open(DATASET_DIR[dataset_name] + "/mappings/" + entity_name + 'id2dbid.txt', 'w+') as file:
                header = ["kgid", "dbid", "dblink"]
                file.write("\t".join(header) + "\n")
                for new_id, old_id in new_id2old_id.items():
                    entity_dblink = entity_id2dblink[old_id]
                    file.write(str(new_id) + '\t' + str(old_id) + '\t' + entity_dblink + "\n")
            file.close()

            # Zip entities
            zip_file(filename)

    def generate_kg_relations(self):
        dataset_name = args.dataset
        mappings = get_all_entity_mappings(dataset_name)

        no_of_movies = len(mappings['movie'])+1
        movie_id_entity = edict(
            producer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/produced_by_producer_m_pr.txt'),
            distributor=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/distributor_m_dis.txt'),
            writer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/wrote_by_m_w.txt'),
            cinematographer=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/cinematography_m_ci.txt'),
            category=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/belong_to_m_ca.txt'),
            actor=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/starring_m_a.txt'),
            director=([[] for _ in range(no_of_movies)], DATASET_DIR[dataset_name] + '/relations/directed_by_m_d.txt'),
        )
        relations_path = DATASET_DIR[dataset_name] + "/relations/"
        if not os.path.isdir(relations_path):
            os.makedirs(relations_path)
        invalid = 0
        print("Inserting relations inside buckets...\n")
        with open(KG_COMPLETATION_DATASET_DIR[dataset_name] + '/kg_final.txt', 'r') as file:
            csv_reader = csv.reader(file, delimiter='\n')
            next(csv_reader, None)
            for row in csv_reader:
                row = row[0].strip().split("\t")
                db_pid = int(row[0])
                if db_pid not in mappings['movie']:
                    invalid += 1
                    continue
                head = mappings['movie'][db_pid][0] #id of the movie in the kg
                tail = int(row[2])
                relation = int(row[1])

                if relation not in SELECTED_RELATIONS[dataset_name]:
                    invalid += 1
                    continue
                tail_entity_name = get_tail_entity_name(dataset_name, relation)
                if tail not in mappings[tail_entity_name]:
                    invalid += 1
                    continue
                kg_id_tail = mappings[tail_entity_name][tail]
                movie_id_entity[tail_entity_name][0][head].append(kg_id_tail)
        file.close()

        print("Invalid relationships:", invalid)
        for entitity_name in get_entities_without_user(dataset_name):
            if entitity_name == 'movie': continue
            relationship_filename = movie_id_entity[entitity_name][1]
            associated_entity_list = movie_id_entity[entitity_name][0]
            print("Populating " + relationship_filename + "...\n")
            with open(relationship_filename, 'w+') as file:
                for entitylist_for_movie in associated_entity_list:
                    s = ' '.join([str(entitity) for entitity in entitylist_for_movie])
                    file.writelines(s)
                    file.write("\n")
            zip_file(relationship_filename)

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=ML100K, help='One of {ML1M, LASTFM}')
    args = parser.parse_args()

    if args.dataset == ML100K:
        ML100KDatasetMapper(args)
    # elif args.dataset == LASTFM:
    #     LastFmDatasetMapper(args)
    # elif args.dataset == ML100K:
    #     ML100kDatasetMapper(args)
    else:
        print("Invalid dataset string, chose one between [ml1m, lastfm]")
