import numpy as np

def back_transform(y_log, y_hat_log, pk_param):
    y_log = np.array(y_log)
    y_hat_log = np.array(y_hat_log)
    
    if pk_param == 'f':
        # 130 is the max bioavailability
        y = (10 ** y_log / (1 + 10 ** y_log)) * 130
        y_hat = (10 ** y_hat_log / (1 + 10 ** y_hat_log)) * 130
    else:
        y = 10 ** y_log
        y_hat = 10 ** y_hat_log
    
    return y, y_hat

def back_transform_predict(y_hat_log, pk_param):
    if pk_param == 'f':
        # 130 is the max bioavailability
        y_hat = (10 ** y_hat_log / (1 + 10 ** y_hat_log)) * 130
    else:
        y_hat = 10 ** y_hat_log
    
    return y_hat

def gmfe(y_log, y_hat_log, pk_param):
    y, y_hat = back_transform(y_log, y_hat_log, pk_param)
    return 10 ** (np.sum(np.abs(np.log10(y_hat / y))) / len(y))

def rmsle(y_log, y_hat_log):
    y_log = np.array(y_log)
    y_hat_log = np.array(y_hat_log)
    return np.sqrt(np.mean((y_log - y_hat_log) ** 2))

def afe(y_log, y_hat_log, pk_param):
    y, y_hat = back_transform(y_log, y_hat_log, pk_param)
    return 10 ** (np.sum(np.log10(y_hat / y)) / len(y))

def pearson_r(y_log, y_hat_log):
    y_log = np.array(y_log)
    y_hat_log = np.array(y_hat_log)
    cov = np.mean((y_log - np.mean(y_log)) * (y_hat_log - np.mean(y_hat_log)))
    return cov / (np.std(y_log) * np.std(y_hat_log))

if __name__ == '__main__':
    pass
