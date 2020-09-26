# 用于评估模型效果
import json
import math

f = open('test_result.json', 'r')
data = json.loads(f.read())
f.close()


def get_diff_vector(entry):
    res = {'pitch': 0.00, 'yaw': 0.00, 'roll': 0.00}
    res['pitch'] += abs(float(entry['expected']['pitch']) - float(entry['actual']['pitch']))
    res['roll'] += abs(float(entry['expected']['roll']) - float(entry['actual']['roll']))
    res['yaw'] += abs(float(entry['expected']['yaw']) - float(entry['actual']['yaw']))
    return res


def mae():
    sum = 0
    res = {'pitch': 0.00, 'yaw': 0.00, 'roll': 0.00}
    for video in data:
        for frame in data[video]:
            entry = data[video][frame]
            if entry['expected'] is not None:
                sum += 1
                diff = get_diff_vector(entry)
                res['pitch'] += diff['pitch']
                res['roll'] += diff['roll']
                res['yaw'] += diff['yaw']
    for angle in res:
        res[angle] /= sum
    return res, sum


def mse():
    avg, sum = mae()
    print('MAE', avg)
    res = {'pitch': 0.00, 'yaw': 0.00, 'roll': 0.00}
    for video in data:
        for frame in data[video]:
            entry = data[video][frame]
            if entry['expected'] is not None:
                diff = get_diff_vector(entry)
                res['pitch'] += pow(diff['pitch']-avg['pitch'], 2)
                res['roll'] += pow(diff['roll'] - avg['roll'], 2)
                res['yaw'] += pow(diff['yaw'] - avg['yaw'], 2)
    for angle in res:
        res[angle] /= sum
    return res


def vector_mae():
    sum = 0
    res = 0.00
    for video in data:
        for frame in data[video]:
            entry = data[video][frame]
            if entry['expected'] is not None:
                sum += 1
                diff = get_diff_vector(entry)
                v = 0.00
                for angle in diff:
                    v += pow(diff[angle], 2)
                v = math.sqrt(v)
                res += v
    return res/sum, sum


def vector_mse():
    avg, sum = vector_mae()

    print('V-MAE:', avg)
    res = 0.00
    for video in data:
        for frame in data[video]:
            entry = data[video][frame]
            if entry['expected'] is not None:
                diff = get_diff_vector(entry)
                v = 0.00
                for angle in diff:
                    v += pow(diff[angle], 2)
                v = math.sqrt(v)
                res += pow(v-avg, 2)
    return res / sum


print('MSE:', mse())
print('V-MSE:', vector_mse())
