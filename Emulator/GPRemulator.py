import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GridSearchCV
import joblib
from scipy.optimize import fmin_l_bfgs_b

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, TransformerMixin


def load_emulator_data(f):
    """
    [[label],[[data_vector0],[data_vector2],...]]
    """
    data = []
    with open(f, "r") as file:
        for line in file:
            parts = line.strip().split(":")
            label = [float(num) for num in parts[0].split(",")]
            values = []

            for i in range(1,len(parts)-1):
                values.append([float(num) for num in parts[i].split(",")])

            data.append([label,values])
    return data


def optimizer(obj_func, initial_theta, bounds, maxiter=int(1e7)):

    result = fmin_l_bfgs_b(func=obj_func, x0=initial_theta, bounds=bounds, maxiter=maxiter)
    return result[0], result[1]

def custom_optimizer(obj_func, initial_theta, bounds):
    return optimizer(obj_func, initial_theta, bounds, maxiter=int(1e7))


def GPR_search(input, out, param_dist):

    X_train, y_train = input, out
    
    gpr = GaussianProcessRegressor(random_state=42)

    random_search = GridSearchCV(estimator=gpr, param_grid=param_dist, scoring='r2', cv=5, n_jobs=160, verbose=1)
    
    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_params_
    

    best_gpr = GaussianProcessRegressor(kernel=best_params['kernel'], n_restarts_optimizer=best_params['n_restarts_optimizer'], normalize_y=best_params['normalize_y'], optimizer=best_params['optimizer'], random_state=42)
    best_gpr.fit(X_train, y_train)
    
    return best_params, best_gpr

def GPR_search_trans(input, out, param_dist):
    X_train, y_train = input, out
    
    # 定义QuantileTransformer和GaussianProcessRegressor
    transformer = QuantileTransformer()
    gpr = GaussianProcessRegressor(random_state=42)

    # 定义TransformedTargetRegressor
    ttr = TransformedTargetRegressor(regressor=gpr, transformer=transformer)

    # 使用GridSearchCV进行参数搜索
    grid_search = GridSearchCV(estimator=ttr, param_grid=param_dist, scoring='r2', cv=5, n_jobs=160, verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_

    # print(best_params)
    
    # 提取最佳参数
    best_regressor_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('regressor__')}
    best_transformer_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('transformer__')}
    
    # 创建并训练最佳模型
    best_transformer = QuantileTransformer(**best_transformer_params)
    best_gpr = GaussianProcessRegressor(**best_regressor_params, random_state=42)
    best_ttr = TransformedTargetRegressor(regressor=best_gpr, transformer=best_transformer)
    best_ttr.fit(X_train, y_train)
    
    return best_params, best_ttr

def pca(data, dim):
    pca = PCA(n_components=dim)
    pca.fit(data)
    data_pca = pca.transform(data)
    eigen = pca.components_
    mean = pca.mean_
    var = pca.explained_variance_ratio_

    return data_pca, eigen, mean, var

class PCAForY(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, fid_mean=None):
        self.n_components = n_components
        self.fid_mean = fid_mean
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        if self.fid_mean is None:
            self.pca.fit(X)
        else:
            X_centered = X-self.fid_mean
            self.pca.fit(X_centered)
        return self

    def transform(self, X, y=None):
        if self.fid_mean is None:
            transformed_X = self.pca.transform(X)
        else:
            X_centered = X-self.fid_mean
            transformed_X = self.pca.transform(X_centered)
        return transformed_X

    def inverse_transform(self, X, y=None):
        if self.fid_mean is None:
            inv_trans_X = self.pca.inverse_transform(X)
        else:
            inv_trans_X = self.pca.inverse_transform(X) + self.fid_mean
        return inv_trans_X

class Emulator:
    def __init__(self, transformer=None, regressor=None):
        self.model = None
        self.transformer = transformer
        self.regressor = regressor
        self.cov = None

    def fit(self, X, y):
        if self.model is not None:
            print("A pre-trained model is provided. Skipping training.")
            return
        else:

            # 如果 transformer 为 None，则不使用任何转换器
            transformer = self.transformer

            # 默认使用GPR regressor
            if self.regressor is None:
                regressor = GaussianProcessRegressor()
            else:
                regressor = self.regressor
                
            self.model = TransformedTargetRegressor(
                regressor=regressor,
                transformer=transformer,
                check_inverse=False
                )
            # 拟合模型
            print("Training...")
            self.model.fit(X, y)

    def predict(self, X, require_cov=False):
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please call `fit` before `predict`.")
        # Todo: require_cov can not deal with multiple inputs 
        if require_cov:
            if hash(type(self.regressor)) != hash(type(GaussianProcessRegressor())):
                print("Warning: Covariance is only available for GaussianProcessRegressor. Returning None.")
                self.cov = None
                return (self.model.predict(X)).squeeze()
            regressor = self.model.regressor_
            if self.transformer is None:
                _, cov_pred = regressor.predict(X, return_cov=require_cov)
                cov = np.diag(cov_pred.flatten())
            else:
                mean_pred, cov_pred = regressor.predict(X, return_cov=require_cov)
                transformer = self.model.transformer_

                def inverse_transform_cov(y_trans, cov_trans, transformer, epsilon=1e-10):
                    # Number of dimensions
                    n_dim = y_trans.shape[0]

                    # Transform y_trans to original space to get output shape
                    y_orig = transformer.inverse_transform(y_trans.reshape(1, -1)).flatten()
                    orig_dim = y_orig.shape[0]

                    # Initialize Jacobian matrix
                    jacobian = np.zeros((orig_dim, n_dim))

                    # Compute Jacobian using finite differences
                    for i in range(n_dim):
                        y_trans_plus = np.copy(y_trans)
                        y_trans_minus = np.copy(y_trans)
                        y_trans_plus[i] += epsilon
                        y_trans_minus[i] -= epsilon

                        y_orig_plus = transformer.inverse_transform(y_trans_plus.reshape(1, -1)).flatten()
                        y_orig_minus = transformer.inverse_transform(y_trans_minus.reshape(1, -1)).flatten()

                        jacobian[:, i] = (y_orig_plus - y_orig_minus) / (2 * epsilon)

                    # Transform the covariance matrix using the Jacobian
                    cov_orig = jacobian @ cov_trans @ jacobian.T
                    return cov_orig

                cov = inverse_transform_cov(mean_pred.flatten(), np.diag(cov_pred.flatten()), transformer)

            self.cov = cov
        else:
            # set self.cov = None to update the cov at next prediction
            self.cov = None

        return (self.model.predict(X)).squeeze()
    
    def get_covriance(self):
        if self.cov is None:
            print("Warning: Covariance has not been calculated yet. Returning np.inf.")
            return np.inf  # Placeholder for covariance
        return self.cov

    def save(self, file_path):
        if self.model is None:
            raise ValueError("The model has not been trained yet. Please call `fit` before saving.")
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path):
        # update the status of self.cov before load a new model
        self.cov = None
        self.model = joblib.load(file_path)
        self.transformer = self.model.transformer
        self.regressor = self.model.regressor
        print(f"Model loaded from {file_path}")

if __name__ == "__main__":

    dataset = load_emulator_data("/home/ljy/BettiCurveCosmo/Data/EmulatorData/nwLH_emulator_dimensionless_[(1,6),(3,12),(9,16)].bc")
    input = np.array([np.array(d[0]) for d in dataset], dtype=np.float32)
    output = [np.array([d[1][dim] for d in dataset], dtype=np.float32) for dim in range(3)]

    param_grid = {
        "transformer__n_components": [5],
        "regressor__n_restarts_optimizer": [0, 15, 45, 60],
        "regressor__normalize_y": [False],
        "regressor__kernel": [
            Matern(length_scale=np.ones_like(input[0]), nu=2.5, length_scale_bounds=(1e-8, 1e7)),
            Matern(length_scale=np.ones_like(input[0]), nu=1.5, length_scale_bounds=(1e-8, 1e7)),
            Matern(length_scale=np.ones_like(input[0]), nu=0.5, length_scale_bounds=(1e-8, 1e7))
        ],
        "regressor__alpha": [1e-6, 1e-7],
        "regressor__optimizer": [custom_optimizer]
    }

    for i in range(3):
        y_train = output[i]
        transformer = PCAForY()
        gpr = GaussianProcessRegressor(random_state=42)
        # 使用 TransformedTargetRegressor 包装 Pipeline 以对 y 进行变换
        model = TransformedTargetRegressor(
            regressor=gpr,
            transformer=transformer
        )
        # Create GridSearchCV with the pipeline and the parameter grid
        search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5, n_jobs=64, verbose=5)
        search.fit(input, y_train)
        best_params = search.best_params_
        best_model = search.best_estimator_
        # Save the best model
        try:
            joblib.dump(best_model, f'/home/ljy/project/emulator/GPR/model/exp2/search14/gpr{i}.pkl')
        except:
            pass
        print(f"Best parameters for output {i}: {best_params}")

