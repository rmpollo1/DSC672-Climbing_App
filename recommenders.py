import os 
import numpy as np 
from surprise.dump import load
import tensorflow as tf
import pickle

# Set Tensorflow Logging to Error Only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Directory To Recommender Files / Directories
RECOMMENDERS_DIR = os.getcwd() + '/recommenders/'
AUTOREC = RECOMMENDERS_DIR + 'AutoRecNew'
CASER = RECOMMENDERS_DIR + 'CASER'
CASER7 = RECOMMENDERS_DIR + 'CASER7'

class SurpriseRecommender(object):
    def __init__(self):
        super().__init__()
    
    def load_recommender(self,recommender):
        '''
        Load pickled model from recommender directory

        Param 
        ------
        recommender: str
            - accepts 'knn' or 'svd' 
        '''
        if recommender == 'knn':
            _, model = load(RECOMMENDERS_DIR + 'knn_recommender.pickle')
        elif recommender == 'svd':
            _, model = load(RECOMMENDERS_DIR + 'mf_recommender.pickle')
        else:
            model = None
        
        self.model_type = recommender
        self.model = model

        self.set_data(self.model)

    def load_from_path(self,recommender_path,model_type=None):
        '''
        Load pickled surprise recommender from supplied file path.

        Param
        ------ 
        recommender_path: str
            - path to recommender
        model_type: str (Optional)
            - type of model
            - default (None)
        '''
        _, model = load(recommender_path)

        self.model = model
        self.model_type = model_type 

        self.set_data(self.model)

    def set_data(self,model):
        '''
        Sets Recommenders internal dataset
        which contains mappings from raw ids to inner ids.
        '''
        self.dataset = model.trainset 

        self.raw_iids = np.array([self.dataset.to_raw_iid(item) for item in self.dataset.all_items()])
        self.raw_uids = np.array([self.dataset.to_raw_uid(item) for item in self.dataset.all_users()])


    def predict_items(self, uid, iids,order=None):
        '''
        Predicts item ratings for a single user id 
        and multiple item ids. Can be returned in order.

        Params
        -------
        uid: 
            - user id
        iids: np.array
            - array of user ids
        order: str (Optional)
            - default (None)
            - accepted
                - ASCENDING
                - DESCENDING

        Returns
        -------
        tuple (iids: np.array, estimated_ratings: np.array)
        '''
        est_ratings = np.empty(len(iids))
        for idx, iid in enumerate(iids):

            _, _, _, est, info = self.model.predict(uid,iid)

            est_ratings[idx] = est

        if order:
            if order == 'ASCENDING':
                step = 1
            if order == 'DESCENDING':
                step = -1
            idx_order = np.argsort(est_ratings)[::step]
            iids, est_ratings = iids[idx_order], est_ratings[idx_order]

        return iids, est_ratings

    def top_k(self, uid, k=10):
        '''
        Returns the top-k items from the training data

        Params
        ------
        uid:
            - User id
        k: int
            - Number of items to return 

        Returns 
        -------
        tuple (item_ids: np.array, est_ratings: np.array)

        Notes
        ------
        est_ratings are unclipped they can fall out of the rating scale
        '''
        all_ratings = np.empty(len(self.raw_iids))
        for idx, iid in enumerate(self.raw_iids):
            _,_,_,est,_ = self.model.predict(uid,iid,clip=False)
            all_ratings[idx] = est
        
        top_k = np.argsort(all_ratings)[::-1][:k]

        return self.raw_iids[top_k], all_ratings[top_k]

    def knows_user(self,uid):
        '''
        Checks if user is present in the training data

        Params
        -------
        user id

        Returns
        --------
        bool
        '''
        return np.any(self.raw_uids == uid)
    def knows_item(self,iid):
        '''
        Checks if item is present in the training data

        Params
        -------
        item id

        Returns
        -------
        bool
        '''
        return np.any(self.raw_iids == iid)

    def sample(self,item_user='users',k=10):
        '''
        Randomly samples users or items 
        that are present in the data the model
        was fitted on. 

        Params 
        ------
        item_user: str
            - either sample known items or users
            - default (users)
            - accepted values
                - users
                - items 
        k: int
            - number to items/users to return 

        Returns 
        --------
        np.array
            - shape (k,)
        '''
        if item_user == 'items':
            return np.random.choice(self.raw_iids,k,replace=False)
        else:
            return np.random.choice(self.raw_uids,k,replace=False)

class AutoRecWrapper(object):
    def __init__(self):
        super().__init__()

    def load_model(self,path_to_model):
        self.model = tf.saved_model.load(path_to_model)
        with open(path_to_model+'/assets/encodings.pb','rb') as f:
            # Dictionary of Dictionaries
            # Keys:
            # - user_encoder: maps raw uid to inner uid
            # - user_decoder: maps inner uid to raw uid
            # - item_encoder: same as above for iid
            # - item_decoder: same as above for iid
            self.mappings = pickle.load(f)

        self.uids = np.array([k for k in self.mappings['user_encoder'].keys()])
        self.iids = np.array([k for k in self.mappings['item_encoder'].keys()])

        self.n_users = self.uids.shape[0]
        self.n_items = self.iids.shape[0] 


        indices = np.genfromtxt(path_to_model+'/assets/rating_indices.txt')
        values = np.genfromtxt(path_to_model+'/assets/rating_values.txt')
        self.sparse_ratings = tf.SparseTensor(indices,tf.cast(values,tf.float32),[self.n_users,self.n_items])
        self.global_mean = values.mean()

        

    def predict(self,uid,iids,order=None):
        '''
        Predicts estimated ratings for user id
        for item ids supplied

        Params
        ------
        uid: int
        iids: np.array
            - shape (None,)
            - 
        order: str
            - Ordering of returns iids and est. ratings
            - Default: None
            - Accepted Values
                - DESCENDING
                - ASCENDING

        Returns
        -------
        tuple (np.array,np.array)
        - returns iids and estimated ratings as tuple
        - in order specified by order parameter
        '''
        inner_uid = self.mappings['user_encoder'].get(uid)

        if not inner_uid:
            # If user is not know return global mean of ratings
            pred_ratings = np.array([self.global_mean for _ in iids])
            return iids, pred_ratings

        # Check for unknown item ids
        unknown_iids = np.setdiff1d(iids,self.iids)
        if unknown_iids.size:
            known_iids = np.setdiff1d(iids,unknown_iids)
            unknown_rating = [self.global_mean for _ in unknown_iids]
        else:
            known_iids = iids

        # Concat Full list of item ids
        full_iids = np.concatenate((known_iids,unknown_iids),0)

        # Map known item ids to inner ids
        inner_iids = [self.mappings['item_encoder'][iid] for iid in known_iids]

        # Get sparse user row of ratings dataset
        past_ratings = tf.sparse.slice(self.sparse_ratings,[inner_uid,0],[1,self.n_items])
        # Predict all items ratings then gather only requested item ids ratings
        pred_ratings = tf.gather(tf.squeeze(self.model.predict(past_ratings)),inner_iids) 

        # Add unknown item id ratings to model predictions
        # If there are any 
        if unknown_iids.size:
            pred_ratings = tf.concat([pred_ratings,unknown_rating],0)

        # Sort items ids and estimated ratings if provided
        if order:
            ordered_idx = tf.argsort(pred_ratings,direction=order)
            return full_iids[ordered_idx], pred_ratings.numpy()[ordered_idx]
        else: 
            return full_iids, pred_ratings.numpy()

    def top_k(self,uid,k=10):
        '''
        Returns the top-k items from the training data

        Params
        ------
        uid:
            - User id
        k: int
            - Number of items to return 

        Returns 
        -------
        tuple (item_ids: np.array, est_ratings: np.array)

        Notes
        ------
        est_ratings are unclipped they can fall out of the rating scale
        '''
        if self.known_user(uid):
            inner_uid = self.mappings['user_encoder'][uid]
            past_ratings = tf.sparse.slice(self.sparse_ratings,[inner_uid,0],[1,self.n_items])
            pred_ratings = tf.squeeze(self.model.predict(past_ratings))
            top_k_idx = tf.argsort(pred_ratings,direction='DESCENDING')[:k]
            top_k_ratings = tf.gather(pred_ratings,top_k_idx)
            top_k_iids = np.array([self.mappings['item_decoder'][iid] for iid in top_k_idx.numpy()])
            return top_k_iids, top_k_ratings.numpy()
        else:
            # If Unknown user randomly sample items and return with global mean rating
            return self.sample_items(k), np.array([self.global_mean for _ in range(k)])

    # Randomly Sample w/o replacement 
    # raw user / item ids 
    def sample_users(self,k=10):
        return np.random.choice(self.uids,k,replace=False)
    def sample_items(self,k=10):
        return np.random.choice(self.iids,k,replace=False)

    # Tests membership of item or user ids
    def known_user(self,uid):
        return uid in self.mappings['user_encoder']
    def knows_item(self,iid):
        return iid in self.mappings['item_encoder']
    
class CaserWrapper(object):
    def __init__(self):
        super().__init__()

    def load_from_path(self,path_to_model):
        self.model = tf.saved_model.load(path_to_model)
        with open(path_to_model+'/assets/mapping.pickle','rb') as f:
            # Dictionary of Dictionaries
            # Keys:
            # - user_encoder: maps raw uid to inner uid
            # - user_decoder: maps inner uid to raw uid
            # - item_encoder: same as above for iid
            # - item_decoder: same as above for iid
            self.mappings = pickle.load(f)

        self.uids = np.array([k for k in self.mappings['user_encoder'].keys()])
        self.iids = np.array([k for k in self.mappings['item_encoder'].keys()])

        self.inner_iids = np.array([k for k in self.mappings['item_decoder'].keys()])

        self.n_users = self.uids.shape[0]
        self.n_items = self.iids.shape[0] 

        # Input Sequence Length
        self.L = self.model.L.numpy()

    def _transform_seq(self,last_n_iids):
        '''
        Pads or trims last iteraction list to correct size
        to input into the model
        '''
        if last_n_iids.shape[0] < self.L:
            last_n_iids = np.pad(last_n_iids,(self.L-last_n_iids.shape[0],0),constant_values=0)
        elif last_n_iids.shape[0] > self.L:
            last_n_iids = last_n_iids[-self.L:]
        return last_n_iids 
    
    def predict_items(self,uid,last_n_iids,iids,order=None):
        '''
        Predict Ranking for select items

        Params
        ------
        uid: int 
            - Integer for user id
        last_n_iids: np.array[int]
            - Numpy Array of last N iteracted Item Ids
            - sequences less than required will be padded with unknown (0)
            - longer sequence will be trimmed to input lenght
            - Sequence format [t-n, ..., t-1, t]
        iids: np.array[int]
            - Numpy Array of Item Ids to rank
        order: str (Optional)
            - default: None
            - Accepted Values
                - DESCENDING
                - ASCENDING
            - should items be returned in order

        Returns 
        -------
        (iids,ranking): tuple(np.array,np.array)
        '''
        inner_uid = self.mappings['user_encoder'].get(uid,0)
        inner_uid = tf.reshape(inner_uid,(1,1))



        past_inner_iids = tf.expand_dims([
            self.mappings['item_encoder'].get(iid,0)
            for iid in self._transform_seq(last_n_iids)
        ],0)


        inner_iids = tf.expand_dims([
            self.mappings['item_encoder'].get(iid,0)
            for iid in iids
        ],0)

        est_rating = tf.squeeze(
            # Predict Ranking for provided items
            self.model.predict_with_iids(
                inner_uid,past_inner_iids,inner_iids,
                tf.constant(False,tf.bool))
          )

        if order:
            # Order Ranking
            order_idx = tf.argsort(est_rating,direction=order)
            iids = iids[order_idx.numpy()] 
            order_rating = tf.gather(est_rating,order_idx)
            return iids, order_rating.numpy()
        else:
            return iids, est_rating.numpy()

    def top_k(self,uid,last_n_iids,k=10):
        '''
        Predict Top-K items from recommender

        Params
        ------
        uid: int 
            - Integer for user id
        last_n_iids: np.array[int]
            - Numpy Array of last N iteracted Item Ids
            - sequences less than required will be padded with unknown (0)
            - longer sequence will be trimmed to input lenght
            - Sequence format [t-n, ..., t-1, t]
        k: int (Optional)
            - default 10
            - Number of items to return

        Returns 
        -------
        (iids,ranking): tuple(np.array,np.array)
        '''
        inner_uid = self.mappings['user_encoder'].get(uid,0)
        inner_uid = tf.reshape(inner_uid,(1,1))
        
        past_inner_iids = tf.expand_dims([
            self.mappings['item_encoder'].get(iid,0)
            for iid in self._transform_seq(last_n_iids)
        ],0)


        # Predict ranking for all items 
        predictions = self.model.predict_all_iids(inner_uid,past_inner_iids,tf.constant(False,tf.bool))
        # Reduce Prediction to 1d Tensor
        predictions = tf.squeeze(predictions)
        # Get Top-k Items 
        top_k = tf.argsort(predictions,direction='DESCENDING')[:k]

        top_k_iids = np.array([self.mappings['item_decoder'][inner_iid] for inner_iid in top_k.numpy()])
        top_k_ratings = tf.gather(predictions,top_k).numpy()
        return top_k_iids, top_k_ratings 


    def top_k_mask(self,uid,last_n_iids,k=10):
        inner_uid = self.mappings['user_encoder'].get(uid,0)
        inner_uid = tf.reshape(inner_uid,(1,1))

        past_inner_iids = tf.expand_dims([
            self.mappings['item_encoder'].get(iid,0)
            for iid in self._transform_seq(last_n_iids)
        ],0)

        iids_to_predict = tf.cast(
            tf.expand_dims(np.setdiff1d(self.inner_iids,past_inner_iids),0),
            tf.int32
        )
        

        est_rating = tf.squeeze(
            # Predict Ranking for provided items
            self.model.predict_with_iids(
                inner_uid,past_inner_iids,iids_to_predict,
                tf.constant(False,tf.bool))
        )

        order_idx = tf.argsort(est_rating,direction='DESCENDING')[:k]
        top_k_inner_iids = tf.gather(tf.squeeze(iids_to_predict),order_idx)
        top_k_raw_iids = np.array([self.mappings['item_decoder'][iid] for iid in top_k_inner_iids.numpy()])
        order_rating = tf.gather(est_rating,order_idx)
        return top_k_raw_iids, order_rating.numpy()

    # Randomly Sample w/o replacement 
    # raw user / item ids 
    def sample_users(self,k=10):
        return np.random.choice(self.uids,k,replace=False)
    def sample_items(self,k=10):
        return np.random.choice(self.iids,k,replace=False)

    # Tests membership of item or user ids
    def known_user(self,uid):
        return uid in self.mappings['user_encoder']
    def knows_item(self,iid):
        return iid in self.mappings['item_encoder']