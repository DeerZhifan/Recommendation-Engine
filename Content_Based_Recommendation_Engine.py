import random
import pandas as pd
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender():
    #  初始化参数
    def __init__(self):
        #  为用户推荐10部电影
        self.n_rec_movies = 10

        #  训练集与测试集
        self.ratio = 0.7
        self.train = {}
        self.test  = {}

        #  电影元数据
        self.movies = 0
        self.rating = 0
        self.tags   = 0
        self.m_cnt  = 0

        #  电影评分权重计算参数
        self.quantile = 0.95
        self.m = 0
        self.c = 0

        #  电影相似度矩阵
        self.cosine_sim = 0

        print('Recommending movies number: {0}'.format(self.n_rec_movies))

    #  载入数据
    def load_file(self, filepath):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')
        print('Succeed in loading file!')

    #  划分训练集与测试集
    def get_dataset(self, filepath):
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

        print('Succeed in building train and test dataset!')
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

    #  提取电影标签
    def tag_extraction(self, t_csv):
        self.tags = pd.read_csv(t_csv)[['movieId', 'tag']]
        stemmer = SnowballStemmer('english')
        t_dict = dict()

        for m_id in set(self.tags['movieId']):
            tag = self.tags[self.tags['movieId'] == m_id]['tag']
            t_dict[m_id] = str(list(set([stemmer.stem(i) for i in tag])))

        m_tag = pd.DataFrame(t_dict, index=[0]).transpose().reset_index()
        m_tag.columns = ['movieId', 'tag']
        m_tag['tag'] = m_tag['tag'].apply(literal_eval)

        return m_tag

    #  清洗数据，获得电影元数据
    def movie_metadata(self, m_csv, r_csv, t_csv):
        print('Preparing movie metadata...')

        self.movies = pd.read_csv(m_csv)
        self.m_cnt  = len(self.movies)

        self.movies['year']   = self.movies['title'].apply(lambda x: x[-5:-1])
        self.movies['title']  = self.movies['title'].apply(self.reset_title)
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|'))

        self.movies = pd.merge(self.movies, self.rating_cnt_avg(r_csv), how='left', on='movieId')
        self.movies = pd.merge(self.movies, self.tag_extraction(t_csv), how='left', on='movieId')

        self.m = self.movies['r_cnt'].quantile(self.quantile)
        self.movies = self.movies[self.movies['r_cnt'] >= self.m]
        self.c = self.movies['r_avg'].mean()
        self.movies['r_weight'] = self.movies.apply(self.r_weight, axis=1)

        self.movies['tag'] = self.movies['tag'].apply(lambda x: [] if str(x) == 'nan' else x)
        self.movies['description'] = self.movies['genres'] + self.movies['tag']
        self.movies['description'] = self.movies['description'].apply(lambda x: ' '.join(x))

        self.movies = self.movies.sort_values('r_weight', ascending=False)
        print("Succeed in preparing movie metadata!")

    #  计算电影相似度
    def cal_movie_sim(self):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(self.movies['description'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #  电影推荐
    def recommend(self, u_id):
        rec = set()
        m_sim = pd.DataFrame(self.cosine_sim, columns=self.movies['movieId'], index=self.movies['movieId'])
        watched = [int(i) for i in self.train[u_id]]

        for movie in watched:
            if movie in m_sim.columns:
                top_5 = list(m_sim.sort_values(movie, ascending=False).index[1:6])
                for i in top_5:
                    if i not in watched:
                        rec.add(i)

        rec_df = pd.DataFrame(list(rec), columns=['movieId'])
        top_movies = rec_df.merge(self.movies, how='left', on='movieId')
        top_movies = top_movies.sort_values('r_weight', ascending=False)
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
    t_csv = '/Users/vita/Movie_Recommender/input_data/small/tags.csv'
    u_id = '3'
    engine = ContentBasedRecommender()
    engine.get_dataset(r_csv)
    engine.movie_metadata(m_csv, r_csv, t_csv)
    engine.cal_movie_sim()
    print("for userId={0}, 10 movies are recommended as follows:".format(u_id))
    print(engine.recommend(u_id))
    engine.evaluate()