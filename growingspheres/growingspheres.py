#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .utils.gs_utils import generate_inside_ball, get_distances
from itertools import combinations
import numpy as np
import spacy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_random_state



class GrowingSpheres:
    """
    class to fit the Original Growing Spheres algorithm
    
    Inputs: 
    obs_to_interprete: instance whose prediction is to be interpreded
    prediction_fn: prediction function, must return an integer label
    caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the 
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                caps=None,
                n_in_layer=2000,
                first_radius=0.01,
                dicrease_radius=2,
                sparse=True,
                verbose=False,
                text=False,
                continuous_features=None,
                categorical_features=[],
                categorical_values=[],
                feature_variance=None,
                farthest_distance_training_dataset=None,
                probability_categorical_feature=None,
                min_counterfactual_in_sphere=0
                ):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        if text:
            self.y_obs = prediction_fn(obs_to_interprete)
        else:
            self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
        if target_class == None:
            self.target_other = True
            target_class = self.y_obs
        else:
            self.target_other = False
        self.target_class = target_class
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius
        self.sparse = sparse
        self.text = text
        if text == True:
            try:
                self.nlp = spacy.load('en_core_web_md')
                #self.nlp = spacy.load('en_core_web_lg')
            except OSError:
                print("you must install en_core_web_lg from spacy")        
        else:
            self.nlp = None
        self.verbose = verbose
        if text:
            self.continuous_features = []
        else:
            self.continuous_features = continuous_features if continuous_features != None else [range(len(obs_to_interprete))]
        self.categorical_features = categorical_features
        self.categorical_values = categorical_values
        self.feature_variance = feature_variance
        self.farthest_distance_training_dataset = farthest_distance_training_dataset
        self.probability_categorical_feature = probability_categorical_feature
        self.min_counterfactual_in_sphere = min_counterfactual_in_sphere
        
        if int(self.y_obs) != self.y_obs:
            raise ValueError("Prediction function should return a class (integer)")

        
    def find_counterfactual(self):
        """
        Finds the decision border then perform projections to make the explanation sparse.
        """
        ennemies_, radius, iteration = self.exploration()
        ennemies_ = sorted(ennemies_, 
                                 key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))
        closest_ennemy_ = ennemies_[0]
        farthest_ennemy = ennemies_[-1]
        self.e_star = closest_ennemy_
        if self.sparse == True:
            if self.text:
                return ennemies_[0], ennemies_, radius, iteration
            out = self.feature_selection(closest_ennemy_)
        else:
            out = closest_ennemy_
        return out, ennemies_, radius, iteration
    
    
    def exploration(self):
        """
        Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
        """
        n_ennemies_ = 999
        radius_ = self.first_radius

        iteration = 0
        while n_ennemies_ > 0:
            if not self.text:
                first_layer_ = self.ennemies_in_layer_((0, radius_), self.caps, self.n_in_layer, iteration=iteration, reducing_sphere=True)
                n_ennemies_ = first_layer_.shape[0]
                radius_ = radius_ / self.dicrease_radius
                iteration += 1
                if self.verbose == True:
                    print("%d ennemies found in initial sphere. Zooming in..."%n_ennemies_)
            else:
                n_ennemies_ = 0
            
        else:
            if self.verbose == True:
                print("Exploring...")
            iteration = 0
            step_ = (self.dicrease_radius - 1) * radius_/2.0
            
            cnt=0
            while n_ennemies_ <= self.min_counterfactual_in_sphere/2:
                layer = self.ennemies_in_layer_((radius_, radius_ + step_), self.caps, self.n_in_layer, iteration=iteration+1)
                n_ennemies_ = layer.shape[0]
                radius_ = radius_ + step_
                iteration += 1
                cnt += 1
                if self.text: print("iteration ", iteration)
            if self.verbose == True:
                print("Final number of iterations: ", iteration)
        if self.verbose == True:
            print("Final radius: ", (radius_ - step_, radius_))
            print("Final number of ennemies: ", n_ennemies_)
        return layer, radius_, iteration
    
    
    def ennemies_in_layer_(self, segment, caps=None, n=1000, iteration=1, reducing_sphere=False):
        """
        Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
        """
        if self.categorical_features != []:
            if self.farthest_distance_training_dataset is None:
                print("you must initialize a distance for the percentage distribution")
            else:
                # Initialize the percentage of categorical features that are changed 
                percentage_distribution = segment[1]/self.farthest_distance_training_dataset*100
            layer = generate_categoric_inside_ball(center= self.obs_to_interprete, segment=segment, n=n, percentage_distribution=percentage_distribution,
                                            continuous_features=self.continuous_features, categorical_features=self.categorical_features, 
                                            categorical_values=self.categorical_values, feature_variance= self.feature_variance, reducing_sphere=reducing_sphere,
                                            probability_categorical_feature=self.probability_categorical_feature)
        else:
            layer = generate_inside_ball(self.obs_to_interprete, segment, n, text=self.text, m=iteration, nlp=self.nlp, feature_variance=self.feature_variance)
        
        #cap here: not optimal
        if caps != None:
            cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(cap_fn_)(layer)
        preds_ = self.prediction_fn(layer)
        if self.target_other:
            return layer[np.where(preds_ != self.target_class)]
        return layer[np.where(preds_ == self.target_class)]
    
    
    def feature_selection(self, counterfactual):
        """
        Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class
        
        Inputs:
        counterfactual: e*
        """
        if self.verbose == True:
            print("Feature selection...")

        #print("counterfactual : ", counterfactual)
        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete)), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        out = counterfactual.copy()
        reduced = 0
        
        for k in move_sorted:
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete[k]
            if self.target_other:
                if self.prediction_fn(new_enn.reshape(1, -1)) != self.target_class:
                    out[k] = new_enn[k]
                    reduced += 1
            else:
                if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                    out[k] = new_enn[k]
                    reduced += 1
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out

    
    def feature_selection_all(self, counterfactual):
        """
        Try all possible combinations of projections to make the explanation as sparse as possible. 
        Warning: really long!
        """
        if self.verbose == True:
            print("Grid search for projections...")
        for k in range(self.obs_to_interprete.size):
            print('==========', k, '==========')
            for combo in combinations(range(self.obs_to_interprete.size), k):
                out = counterfactual.copy()
                new_enn = out.copy()
                for v in combo:
                    new_enn[v] = self.obs_to_interprete[v]
                if self.target_other:
                    if self.prediction_fn(new_enn.reshape(1, -1)) != self.target_class:
                        print('bim')
                        out = new_enn.copy()
                        reduced = k
                else:
                    if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                        print('bim')
                        out = new_enn.copy()
                        reduced = k
        if self.verbose == True:
            print("Reduced %d coordinates"%reduced)
        return out
    
    
    
class DirectedGrowingSpheres: # Warning: Not finished, do not use
    """
    class to fit the Original Growing Spheres algorithm
    """
    def __init__(self,
                obs_to_interprete,
                prediction_fn,
                target_class=None,
                caps=None,
                n_in_layer=10000, #en vrai n_in_layer depend de rayon pour garantir meme densite?
                first_radius=0.1,
                dicrease_radius=5):
        """
        """
        self.obs_to_interprete = obs_to_interprete
        self.prediction_fn = prediction_fn
        y_class = int(prediction_fn(obs_to_interprete.reshape(1, -1))[0][1] > 0.5) #marche que pour 2 classes là. en vrai sinon il faut prendre l'argmax
        self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))[0][1]
        #if target_class == None:
        #    target_class = 1 - self.y_obs
        #self.target_class = target_class
        self.target_class = 1  - y_class
        
        self.caps = caps
        self.n_in_layer = n_in_layer
        self.first_radius = first_radius
        self.dicrease_radius = dicrease_radius

        
    def find_counterfactual(self): #nul
        ennemies_ = self.exploration()
        closest_ennemy_ = sorted(ennemies_, 
                                 key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0] 
        out = self.feature_selection(closest_ennemy_)
        return out
    
    
    def exploration(self):
        n_ennemies_ = 999
        radius_ = self.first_radius
        iteration = 0
        
        while n_ennemies_ > 0: #initial step idem: on veut une sphère qui n'a pas d'ennemi (pas le plus smart; peut être garder les derniers ennemis serait intelligent... bref
            first_layer_, y_layer_ = self.layer_with_preds(self.obs_to_interprete, radius_, self.caps, self.n_in_layer)
            n_ennemies_ = first_layer_[np.where(y_layer_ > 0.5)].shape[0] #IMPORTANT; PAREIL NE MARCHE QUE POUR CLASSIF BINAIRE. IL FAUDRA REPENSER CA POUR MULTILCASSE
            if iteration != 0:
                radius_ = radius_ / self.dicrease_radius #IMPORTANT: Le Dicrease devrait pouvoir se faire en fonction du nombre d'ennemis trouves: si beaucoup, on peut reduire de plus que si 3 ennemis.
                print("%d ennemies found in initial sphere. Zooming in..."%n_ennemies_)
            iteration += 1
            
        else:
            print("Exploring...")
            iteration = 0
            step_ = 0.1# (self.dicrease_radius - 1) * radius_/10.0 #a chabger
            #initialisation
            center_ = self.obs_to_interprete
            #layer = first_layer_
            layer, y_layer_ = self.layer_with_preds(self.obs_to_interprete, radius_*5, self.caps, self.n_in_layer)
            self.centers = []
            radius_ = radius_ #bluff
            while n_ennemies_ <= 0:                
                gradient = self.get_exploration_direction(layer, y_layer_) #la on fait comme s'il n'y avait qu'un gradient mais il pourrait y en avoir plusieurs
                center_ = center_ + gradient * step_ #idem
                layer, y_layer_ = self.layer_with_preds(center_, radius_*5, self.caps, self.n_in_layer)
                self.centers.append(center_)
                n_ennemies_ = layer[np.where(y_layer_ > 0.5)].shape[0] #proba d'appartenir à la nouvelle classe
                iteration += 1
                
            print("Final number of iterations: ", iteration)
        print("Final radius: ", (radius_ - step_, radius_))
        print("Final number of ennemies: ", n_ennemies_)
        self.centers = np.array(self.centers)
        return layer[np.where(y_layer_ > 0.5)] #proba d'appartenir à la nouvelle classe
    
    
    def layer_with_preds(self, center, radius, caps=None, n=1000):
        """
        prend obs, genere couche dans sphere, et renvoie les probas d'appartenir à target class
        """
        layer = generate_inside_ball(center, (0, radius), n)
        #cap here: pas optimal, ici meme min et meme max pour toutes variables;
        
        if caps != None:
            cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
            layer = np.vectorize(cap_fn_)(layer)
            
        preds = self.prediction_fn(layer)[:, self.target_class]
        return layer, preds
    
    def get_exploration_direction(self, layer, preds):
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression(fit_intercept=True).fit(layer, preds)
        gradient = lr.coef_
        gradient = gradient / sum([x**2 for x in gradient])**(0.5)
        return gradient
    
    def get_exploration_direction2(self, layer, preds):
        return layer[np.where(preds == preds.max())][0] - self.obs_to_interprete
    
    
    
    def feature_selection(self, counterfactual): #checker
        """
        """
        print("Feature selection...")
        move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete)), key=lambda x: x[1])
        move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
        out = counterfactual.copy()
        reduced = 0
        
        for k in move_sorted:
            new_enn = out.copy()
            new_enn[k] = self.obs_to_interprete[k]
            
            if self.prediction_fn(new_enn.reshape(1, -1))[0][self.target_class] > 0.5: #il faut mettre argmax pour multiclasse
                out[k] = new_enn[k]
                reduced += 1
                
        print("Reduced %d coordinates"%reduced)
        return out

 
    
