from meters import ScalarMeter
width_mult_list_all = [1, 2, 3, 4]
topk = [1,5]
def get_meters(phase):
    """util function for meters"""
    meters_all = {}
    for width_mult in width_mult_list_all:
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(
            phase, str(width_mult)))
        for k in topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, str(width_mult)))
        meters_all[str(width_mult)] = meters
    meters = meters_all
    return meters