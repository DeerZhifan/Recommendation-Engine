import random
import pandas as pd


#  基于评分排序的推荐
class MostPopularRecommender():
    #  初始化参数
    def __init__(self):
        #  为用户推荐10部电影
        self.n_rec_movies = 10

        #  训练集与测试集
        self.ratio = 0.7
        self.train = {}
        self.test = {}

        #  电影元数据
        self.movies = 0
        self.rating = 0
        self.m_cnt = 0

        #  电影评分权重计算参数
        self.quantile = 0.95
        self.m = 0
        self.c = 0

        print('Recommending movies number: {0}'.format(self.n_rec_movies))

    #  载入数据
    def load_file(self, filepath):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    #  划分训练集与测试集
    def get_dataset(self, filepath):
        print('Spliting dataset...')
        train_len = 0
        test_len  = 0

        for line in self.load_file(filepath):
            u_id, m_id, rating, timestamp = line.split(',')
            if random.random() < self.ratio:
                self.train.setdefault(u_id, [])
                self.train[u_id].append(m_id)
                train_len += 1
            else:
                self.test.setdefault(u_id, [])
                self.test[u_id].append(m_id)
                test_len += 1

        print('Done!')
        print('Train: {0}, Test: {1}'.format(train_len, test_len))

    #  处理电影名
    def reset_title(self, title):
        if ',' in title:
            #  “Shining, The (1980)”→“The Shining”
            return ' '.join(title[:-7].split(',')[::-1]).strip()
        else:
            #  "Toy Story (1995)"→“Toy Story"
            return title[:-7]

    #  计算评分权重
    def r_weight(self, m_metadata):
        v = m_metadata['r_cnt']
        r = m_metadata['r_avg']
        return v / (v + self.m) * r + self.m / (v + self.m) * self.c

    #  统计电影评分
    def rating_cnt_avg(self, r_csv):
        r_cnt = dict()
        r_avg = dict()
        self.rating = pd.read_csv(r_csv)

        for m_id in set(self.rating['movieId']):
            r_sum = sum(self.rating[self.rating['movieId'] == m_id]['rating'])
            r_len = len(self.rating[self.rating['movieId'] == m_id]['rating'])
            r_cnt[m_id] = r_len
            r_avg[m_id] = round(r_sum / r_len, 3)

        df_r_cnt = pd.DataFrame(r_cnt, index=[0]).transpose().reset_index()
        df_r_avg = pd.DataFrame(r_avg, index=[0]).transpose().reset_index()
        df_r_cnt.columns = ['movieId', 'r_cnt']
        df_r_avg.columns = ['movieId', 'r_avg']

        r_cnt_avg = pd.merge(df_r_cnt, df_r_avg, how='left', on='movieId')

        return r_cnt_avg

    #  清洗数据，获得电影元数据
    def movie_metadata(self, m_csv, r_csv):
        print('Preparing movie metadata...')

        self.movies = pd.read_csv(m_csv)
        self.m_cnt  = len(self.movies)

        self.movies['year']   = self.movies['title'].apply(lambda x: x[-5:-1])
        self.movies['title']  = self.movies['title'].apply(self.reset_title)
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|'))

        self.movies = pd.merge(self.movies, self.rating_cnt_avg(r_csv), how='left', on='movieId')

        self.m = self.movies['r_cnt'].quantile(self.quantile)
        self.movies = self.movies[self.movies['r_cnt'] >= self.m]
        self.c = self.movies['r_avg'].mean()
        self.movies['r_weight'] = self.movies.apply(self.r_weight, axis=1)
        self.movies = self.movies.sort_values('r_weight', ascending=False)

        print("Done!")

    #  推荐电影
    def recommend(self, u_id):
        watched    = [int(i) for i in self.train[u_id]]
        un_watched = list(set(self.movies['movieId']) - set(watched))
        df_uw = pd.DataFrame(un_watched, columns=['movieId'])
        top_movies = pd.merge(self.movies, df_uw, on='movieId')
        rec_movies = top_movies[['movieId', 'title', 'year']][:self.n_rec_movies]
        return rec_movies

    #  推荐评估
    def evaluate(self):
        print('Evaluation start...')

        hit       = 0
        rec_cnt   = 0
        test_cnt  = 0
        m_all_rec = set()

        for u_id in self.train:
            m_test = self.test[u_id]
            m_rec  = self.recommend(u_id)
            m_rec  = list(set(m_rec['movieId']))
            for movie in m_rec:
                if str(movie) in m_test:
                    hit += 1
                m_all_rec.add(movie)
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
    u_id = '3'
    engine = MostPopularRecommender()
    engine.get_dataset(r_csv)
    engine.movie_metadata(m_csv, r_csv)
    print("for userId={0}, 10 movies are recommended as follows:".format(u_id))
    print(engine.recommend(u_id))
    engine.evaluate()

    ###########################################################################
    #  Recommending movies number: 10                                         #
    #  Spliting dataset...                                                    #
    #  Done!                                                                  #
    #  Train: 70686, Test: 30150                                              #
    #  Preparing movie metadata...                                            #
    #  Done!                                                                  #
    #  for userId=3, 10 movies are recommended as follows:                    #
    #  movieId                                          title  year           #
    #  318                           The Shawshank Redemption  1994           #
    #  858                                      The Godfather  1972           #
    #  2959                                        Fight Club  1999           #
    #  260                 Star Wars: Episode IV - A New Hope  1977           #
    #  50                                  The Usual Suspects  1995           #
    #  296                                       Pulp Fiction  1994           #
    #  2571                                        The Matrix  1999           #
    #  1196    Star Wars: Episode V - The Empire Strikes Back  1980           #
    #  1198  Raiders of the Lost Ark (Indiana Jones and the... 1981           #
    #  356                                       Forrest Gump  1994           #
    #  Evaluation start...                                                    #
    #  Precision = 0.1744                                                     #
    #  Recall = 0.0353                                                        #
    #  Coverage = 0.0041                                                      #
    ###########################################################################