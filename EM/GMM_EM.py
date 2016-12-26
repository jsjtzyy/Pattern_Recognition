# This code from scratch part is completed by yingyiz2, ghe10 and gjin7

from PIL import Image
import numpy as np
import time


# Multi-variable Gaussian Function:
def n_dim_gaussian(x_vector, mu, cov):
    D = np.shape(x_vector)[0]
    Y = np.mat(x_vector - mu)
    temp = np.dot(np.dot(Y, np.linalg.inv(cov)), Y.transpose())  # matrix multiplication
    result = 1.0/((2*np.pi)**(D/2) * np.linalg.det(cov)**0.5) * np.exp(-0.5*temp)
    return float(result)


class GMM_EM:
    x = []
    mu = []
    cov = []
    gamma =[]
    w = []
    n = 0
    m = 0
    var_r = []
    var_g = []
    var_b = []

    def __init__(self, m_size, image_name):
        self.im = Image.open(image_name)
        self.pix = self.im.load()
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.m = m_size
        self.n = self.height * self.width
        self.w = np.ones(self.m)/self.m
        self.tmp = np.zeros(self.m)
        self.new_im = Image.new("RGB", (self.width, self.height))

    # generate training set:
    def generate_training_set(self):
        R = []
        G = []
        B = []
        for i in range(self.width):
            for j in range(self.height):
                if self.im.mode == 'RGBA':
                    r, g, b, a = self.pix[i, j]
                if self.im.mode == 'RGB':
                    r, g, b = self.pix[i, j]
                self.x.append(np.array([r, g, b]))
                R.append(r)
                G.append(g)
                B.append(b)
        self.var_r = np.var(R)
        self.var_g = np.var(G)
        self.var_b = np.var(B)
        print("Training data set is established!")

    # initialize mu1, mu2, mu3, cov1, cov2, cov3, w1, w2, w3:
    def initialize_parameters(self):
        for i in range(self.m):
            var_red = 5 * np.random.randn(self.m) + self.var_r
            var_gre = 5 * np.random.randn(self.m) + self.var_g
            var_blu = 5 * np.random.randn(self.m) + self.var_b

        for i in range(self.m):
            self.cov.append(np.diag(np.array([var_red[i], var_gre[i], var_blu[i]])))
        self.cov = np.array(self.cov)

        self.mu = []
        for i in range(self.m):
            mu_tmp = np.random.rand(3) * 255
            self.mu.append(mu_tmp)
        self.mu = np.array(self.mu)
        self.gamma = np.zeros((self.n, self.m))
        print("Parameters Initialization has completed!")

    # Compute gamma:
    def gammaprob(self):
        start = time.clock()
        res_sum = np.zeros(self.m)
        for i in range(self.n):
            total_sum = 0
            for k in range(self.m):
                self.gamma[i][k] = self.w[k] * n_dim_gaussian(self.x[i], self.mu[k], self.cov[k])
                total_sum += self.gamma[i][k]
            self.gamma[i] /= total_sum
            res_sum += self.gamma[i]
        end = time.clock()
        print("e-step running time: ", end - start)
        return res_sum

    # E step:
    def estep(self):
        return self.gammaprob()

    # M step
    def mstep(self, res_sum):
        start = time.clock()
        self.w = res_sum / self.n
        for k in range(self.m):
            self.mu[k] = np.zeros(3)
            for i in range(self.n):
                self.mu[k] += self.x[i] * self.gamma[i][k]
            self.mu[k] /= res_sum[k]
            self.cov[k] = np.diag([0, 0, 0])
            for i in range(self.n):
                vec = np.mat(self.x[i] - self.mu[k])
                self.cov[k] += self.gamma[i][k] * np.dot(vec.transpose(), vec)  # matrix multiplication
            self.cov[k] /= res_sum[k]
        end = time.clock()
        print("m-step running time:", end - start)

    # plot the image
    def plot_image(self):
        for i in range(self.n):
            for k in range(self.m):
                self.tmp[k] = n_dim_gaussian(self.x[i], self.mu[k], self.cov[k])
            xindex = i % self.height
            yindex = int(i / self.height)
            max_mu = self.mu[np.argmax(self.tmp)]
            self.new_im.putpixel((yindex, xindex), (int(max_mu[0]), int(max_mu[1]), int(max_mu[2])))
        self.new_im.show()

    # save the image
    def save_image(self, name):
        self.new_im.save(name)


def __main__():
    gmm = GMM_EM(3, "corgi.png")
    gmm.generate_training_set()
    gmm.initialize_parameters()

    iteration = 6
    for i in range(iteration):
        start = time.clock()
        # E_step
        res_sum = gmm.estep()
        # M_step
        gmm.mstep(res_sum)
        end = time.clock()
        print("iteration %d run time: " % i, end - start)
        start = time.clock()
        gmm.plot_image()
        end = time.clock()
        print("plotting time: ", end - start)
    gmm.save_image("image_cluster.png")
if __name__ == '__main__':
    __main__()


# print("The w parameters: ")
# for k in range(m):
#     print(w[k])
# print("The mu parameters: ")
# for k in range(m):
#     print(mu[k])
# print("The covariance matrix: ")
# for k in range(m):
#     print(cov[k])









