import numpy as np
import matplotlib.pyplot as plt
# 加载 true 和 pred 数据
true_data = np.load('long_term_forecast_ziqi-alabama_TimesNet_epoch10_24_Fusion_TimesNet_station_do_ftM_sl48_ll24_pl24_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0/true.npy')
pred_data = np.load('long_term_forecast_ziqi-alabama_TimesNet_epoch10_24_Fusion_TimesNet_station_do_ftM_sl48_ll24_pl24_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_dtTrue_Exp_0/pred.npy')




print(true_data)