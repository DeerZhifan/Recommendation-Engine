import math
import random
import pandas as pd
from operator import itemgetter


#  基于用户的协同过滤推荐
class UserCF():
    #  初始化参数
    def __init__(self):
        #  推荐10部电影
        self.n_sim_users  = 10
        self.n_rec_movies = 10

        #  划分训练集与测试集
        self.ratio = 0.7
        self.train = {}
        self.test  = {}

        #  电影元数据
        self.movies = 0
        self.m_cnt  = 0

        #  用户相似度矩阵
        self.user_sim_matrix = {}

        print('Recommending {0} movies...'.format(self.n_rec_movies))

    #  载入数据
    def load_file(self, filepath):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    #  划分训练集与测试集
    def split_dataset(self, filepath):
        print('Splitng dataset...')
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
        print('train: {0} test: {1}'.format(train_len, test_len))

    #  处理电影名
    def reset_title(self, title):
        if ',' in title:
            #  “Shining, The (1998)”→“The Shining”
            return ' '.join(title[:-7].split(',')[::-1]).strip()
        else:
            #  “Toy Story (1995)”→"Toy Story"
            return title[:-7]

    #  清洗数据，获得电影元数据
    def movie_metadata(self, m_csv):
        print('Building movie metadata...')
        self.movies = pd.read_csv(m_csv)
        self.m_cnt  = len(self.movies)

        self.movies['year']  = self.movies['title'].apply(lambda x: x[-5:-1])
        self.movies['title'] = self.movies['title'].apply(self.reset_title)

        print('Done!')

    #  计算用户相似度
    def cal_user_sim(self):
        print('Calculating user similarity matrix...')
        #  建立电影-用户倒排表
        movie_user = {}
        for user, movies in self.train.items():
            for movie in movies:
                movie_user.setdefault(movie, set())
                movie_user[movie].add(user)
        #  计算相似度矩阵
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                n_u = len(self.train[u])
                n_v = len(self.train[v])
                self.user_sim_matrix[u][v] = count / math.sqrt(n_u * n_v)

        print("Calculation done!")

    #  recommend()调用该函数匹配推荐电影的title和year信息
    def get_info(self, rank):
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:self.n_sim_users]
        rank_df = pd.DataFrame(rank, columns=['movieId', 'similarity'])
        rank_df['movieId'] = rank_df['movieId'].apply(lambda m_id: int(m_id))
        m_rank = pd.merge(rank_df, self.movies, how='left', on='movieId')
        n_rec_movies = m_rank.sort_values('similarity', ascending=False)[:self.n_rec_movies]
        n_rec_movies = n_rec_movies[['movieId', 'title', 'year']]

        return n_rec_movies

    #  推荐电影
    def recommend(self, u_id):
        rank = {}
        watched = self.train[u_id]
        n_sim_users = sorted(self.user_sim_matrix[u_id].items(), key=itemgetter(1), reverse=True)[:self.n_sim_users]
        for v, w_uv in n_sim_users:
            for movie, rating in self.train[v].items():
                if movie not in watched:
                    rank.setdefault(movie, 0)
                    rank[movie] += w_uv * float(rating)

        m_rec = self.get_info(rank)
        return m_rec

    #  推荐评估
    def evaluate(self):
        print('Evaluation start...')

        hit         = 0
        rec_cnt     = 0
        test_cnt    = 0
        all_rec_cnt = set()

        for u_id in self.train:
            m_test = self.test.get(u_id, {})
            m_rec  = self.recommend(u_id)
            m_rec  = list(set(m_rec['movieId']))
            for m_id in m_rec:
                if str(m_id) in m_test:
                    hit += 1
                all_rec_cnt.add(m_id)
            rec_cnt  += self.n_rec_movies
            test_cnt += len(m_test)

        precision = hit / rec_cnt
        recall    = hit / test_cnt
        coverage  = len(all_rec_cnt) / self.m_cnt
        result = (precision, recall, coverage)

        print('Precision = %.4f\nRecall = %.4f\nCoverage = %.4f' % result)


if __name__ == '__main__':
    m_csv = '/Users/vita/Movie_Recommender/input_data/small/movies.csv'
    r_csv = '/Users/vita/Movie_Recommender/input_data/small/ratings.csv'
    u_id = '3'
    engine = UserCF()
    engine.split_dataset(r_csv)
    engine.movie_metadata(m_csv)
    engine.cal_user_sim()
    print('For the userId = {0}, 10 movies are recommended as follows:'.format(u_id))
    rec_movies = engine.recommend(u_id)
    print(rec_movies)
    engine.evaluate()

    ###########################################################################
    #  Recommending 10 movies...                                              #
    #  Splitng dataset...                                                     #
    #  Done!                                                                  #
    #  Train: 70616, Test: 30220                                              #
    #  Preparing movie metadata...                                            #
    #  Done!                                                                  #
    #  Calculating user similarity matrix...                                  #
    #  Calculation done!                                                      #
    #  for userId=3, 10 movies are recommended as follows:                    #
    #  movieId                                       title   year             #
    #  1214                                          Alien   1979             #
    #  3396                               The Muppet Movie   1979             #
    #  2617                                      The Mummy   1999             #
    #  1215                               Army of Darkness   1993             #
    #  1200                                         Aliens   1986             #
    #  4571               Bill & Ted's Excellent Adventure   1989             #
    #  34                                             Babe   1995             #
    #  3438                   Teenage Mutant Ninja Turtles   1990             #
    #  1148           Wallace & Gromit: The Wrong Trousers   1993             #
    #  45722                 The Rocky Horror Picture Show   1975             #
    #  Evaluation start...                                                    #
    #  Precision = 0.3113                                                     #
    #  Recall = 0.0626                                                        #
    #  Coverage = 0.0499                                                      #
    ###########################################################################
