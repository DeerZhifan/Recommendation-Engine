import math
import random
import pandas as pd
from operator import itemgetter


#  基于物品的协同过滤推荐
class ItemCF():
    #  初始化参数
    def __init__(self):
        #  推荐10部电影
        self.n_sim_movies = 10
        self.n_rec_movies = 10

        #  划分训练集与测试集
        self.ratio = 0.7
        self.train = {}
        self.test  = {}

        #  电影元数据
        self.movies = 0
        self.m_cnt  = 0

        #  电影相似度度矩阵
        self.m_sim_matrix = {}

        print('Recommending {0} movies'.format(self.n_rec_movies))

    #  载入数据
    def load_file(self, filepath):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    #  划分训练集与测试集
    def split_dataset(self, filepath):
        print('Spliting dataset...')
        train_len = 0
        test_len  = 0

        for line in self.load_file(filepath):
            u_id, m_id, rating, timestamp = line.split(',')
            if random.random() < self.ratio:
                self.train.setdefault(u_id, {})
                self.train[u_id][m_id] = rating
                train_len += 1
            else:
                self.test.setdefault(u_id, {})
                self.test[u_id][m_id] = rating
                test_len += 1

        print('Done!')
        print('Train: {0}, Test: {1}'.format(train_len, test_len))

    #  处理电影名
    def reset_title(self, title):
        if ',' in title:
            return ' '.join(title[:-7].split(',')[::-1]).strip()
        else:
            return title[:-7]

    #  清洗数据，获得电影元数据
    def movie_metadata(self, m_csv):
        print('Building movie metadata...')
        self.movies = pd.read_csv(m_csv)
        self.m_cnt = len(self.movies)

        self.movies['year']  = self.movies['title'].apply(lambda x: x[-5:-1])
        self.movies['title'] = self.movies['title'].apply(self.reset_title)

        print('Done!')

    #  计算电影相似度
    def cal_m_sim(self):
        print('Calculating movie similarity matrix...')
        #  计算相似度矩阵
        m_popularity = {}
        for user, movies in self.train.items():
            for m1 in movies:
                m_popularity.setdefault(m1, 0)
                m_popularity[m1] += 1
                for m2 in movies:
                    if m1 == m2:
                        continue
                    self.m_sim_matrix.setdefault(m1, {})
                    self.m_sim_matrix[m1].setdefault(m2, 0)
                    self.m_sim_matrix[m1][m2] += 1

        for m1, m1_related in self.m_sim_matrix.items():
            for m2, count in m1_related.items():
                self.m_sim_matrix[m1][m2] = count / math.sqrt(m_popularity[m1] * m_popularity[m2])
        print('Done!')

    #  匹配电影信息
    def get_info(self, rank):
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:self.n_sim_movies]
        rank_df = pd.DataFrame(rank, columns=['movieId', 'similarity'])
        rank_df['movieId'] = rank_df['movieId'].apply(lambda x: int(x))
        m_rec = pd.merge(rank_df, self.movies, how='left', on='movieId')
        m_rec = m_rec[['movieId', 'title', 'year']]

        return m_rec

    #  推荐电影
    def recommend(self, u_id):
        watched = self.train[u_id]
        rank = {}
        for m_id, rating in watched.items():
            n_sim_movies = sorted(self.m_sim_matrix[m_id].items(), key=itemgetter(1), reverse=True)[:self.n_sim_movies]
            for m, similarity in n_sim_movies:
                if m not in watched:
                    rank.setdefault(m, 0)
                    rank[m] += similarity * float(rating)
        m_rec = self.get_info(rank)

        return m_rec

    #  推荐评估
    def evaluate(self):
        print('Evaluation start...')

        hit      = 0
        rec_cnt  = 0
        test_cnt = 0
        m_all_rec = set()

        for u_id in self.train:
            m_test = self.test.get(u_id, {})
            m_rec  = self.recommend(u_id)
            m_rec  = list(m_rec['movieId'])
            for m_id in m_rec:
                if str(m_id) in m_test:
                    hit += 1
                m_all_rec.add(m_id)
            rec_cnt  += self.n_rec_movies
            test_cnt += len(m_test)

        precision = hit / rec_cnt
        recall    = hit / test_cnt
        coverage  = len(m_all_rec) / self.m_cnt
        result = (precision, recall, coverage)

        print('Precision = %.4f\nRecall = %.4f\nCoverage = %.4f' % result)


if __name__ == '__main__':
    m_csv = '/Users/vita/Movie_Recommender/input_data/small/movies.csv'
    r_csv = '/Users/vita/Movie_Recommender/input_data/small/ratings.csv'
    u_id  = '3'
    engine = ItemCF()
    engine.split_dataset(r_csv)
    engine.movie_metadata(m_csv)
    engine.cal_m_sim()
    print('For the userId = {0}, 10 movies are recommended as follows:'.format(u_id))
    print(engine.recommend(u_id))
    engine.evaluate()

    ###########################################################################
    #  Recommending 10 movies...                                              #
    #  Splitng train and test dataset...                                      #
    #  Done!                                                                  #
    #  Train: 70616, Test: 30220                                              #
    #  Preparing movie metadata...                                            #
    #  Done!                                                                  #
    #  Calculating movie similarity matrix...                                 #
    #  Calculation done!                                                      #
    #  for userId=3, 10 movies are recommended as follows:                    #
    #  movieId                                       title  year              #
    #  870                                    Gone Fishin'  1997              #
    #  1839                                       My Giant  1998              #
    #  2368                                King Kong Lives  1986              #
    #  3527                                       Predator  1987              #
    #  2661                       It Came from Outer Space  1953              #
    #  4496                                         D.O.A.  1988              #
    #  6371                                 Pokémon Heroes  2003              #
    #  7040                        To Live and Die in L.A.  1985              #
    #  3648                         The Abominable Snowman  1957              #
    #  4942                           The Angry Red Planet  1959              #
    #  Evaluation start...                                                    #
    #  Precision = 0.2846                                                     #
    #  Recall = 0.0575                                                        #
    #  Coverage = 0.0818                                                      #
    ###########################################################################
