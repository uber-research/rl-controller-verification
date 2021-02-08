import matplotlib.pyplot as plt
import numpy as np
import numba as nb

AllProps_jitspec = [
    ('time_signal', nb.float64[:]),
    ('tt', nb.float64[:]),
    ('x', nb.float64[:]),
    ('q', nb.float64[:]),
    ('tset', nb.float64),
    ('tstab', nb.float64),
    ('q_lh', nb.float64),
    ('q_diam', nb.float64),
    ('offset_pct', nb.float64),
    ('overshoot_pct', nb.float64),
    ('time_to_reach_pct', nb.float64),
    ('tt_oob_max', nb.float64),
    ('x_oob_max', nb.float64),
    ('q_oob_max', nb.float64),
    ('tt_oob_min', nb.float64),
    ('x_oob_min', nb.float64),
    ('q_oob_min', nb.float64),
    ('last_t', nb.float64),
    ('bool_agg_lb', nb.float64[:]),
    ('bool_agg_ub', nb.float64[:]),
    ('bool_agg_t', nb.float64[:]),
    ('bool_agg_v', nb.float64[:]),
    ('bool_agg_h', nb.float64[:]),
    ('bool_agg_c', nb.float64[:]),
    ('real_agg_lb', nb.float64[:]),
    ('real_agg_ub', nb.float64[:]),
    ('real_agg_t', nb.float64[:]),
    ('real_agg_v', nb.float64[:]),
    ('real_agg_h', nb.float64[:]),
    ('real_agg_c', nb.float64[:]),

]

@nb.experimental.jitclass(AllProps_jitspec)
class AllProps:
    """
    Observer for property AllProps.
    :param time_signal: nb.float64[:] of nb.float64 signal
    :param tt: nb.float64[:] of nb.float64 signal
    :param x: nb.float64[:] of nb.float64 signal
    :param q: nb.float64[:] of nb.float64 signal
    :param tset: nb.float64 static parameter
    :param tstab: nb.float64 static parameter
    :param q_lh: nb.float64 static parameter
    :param q_diam: nb.float64 static parameter
    :param offset_pct: nb.float64 static parameter
    :param overshoot_pct: nb.float64 static parameter
    :param time_to_reach_pct: nb.float64 static parameter
    """
    def __init__(self, time_signal, tt, x, q, tset, tstab, q_lh, q_diam, offset_pct, overshoot_pct, time_to_reach_pct):
        self.time_signal = time_signal
        assert len(time_signal) == len(tt), 'wrong size for signal tt in AllProps'
        assert len(time_signal) == len(x), 'wrong size for signal x in AllProps'
        assert len(time_signal) == len(q), 'wrong size for signal q in AllProps'
        self.tt = tt
        self.x = x
        self.q = q
        self.tset = tset
        self.tstab = tstab
        self.q_lh = q_lh
        self.q_diam = q_diam
        self.offset_pct = offset_pct
        self.overshoot_pct = overshoot_pct
        self.time_to_reach_pct = time_to_reach_pct

        # oob access tracking
        self.tt_oob_min = np.inf
        self.x_oob_min = np.inf
        self.q_oob_min = np.inf

        # oob access tracking
        self.tt_oob_max = -np.inf
        self.x_oob_max = -np.inf
        self.q_oob_max = -np.inf

        # last evaluation point
        self.last_t = -np.inf

        # last aggregate computation
        self.bool_agg_lb = np.zeros(0)
        self.bool_agg_ub = np.zeros(0)
        self.bool_agg_t = np.array([-np.inf for i in range(0)])
        self.bool_agg_v = np.zeros(0)
        self.bool_agg_h = np.zeros(0)
        self.bool_agg_c = np.zeros(0)

        # last aggregate computation
        self.real_agg_lb = np.zeros(9)
        self.real_agg_ub = np.zeros(9)
        self.real_agg_t = np.array([-np.inf for i in range(9)])
        self.real_agg_v = np.zeros(9)
        self.real_agg_h = np.zeros(9)
        self.real_agg_c = np.zeros(9)

    def apply(self, t):
        assert(self.last_t <= t)
        return self.fun_allProps_0(t)

    def fun_allProps_0(self, t):
        var_staticParam_4 = self.q_lh

        var_realBinop_3 = 0.0 - var_staticParam_4
        var_shiftBoundFwd_2 = t + var_realBinop_3
        var_staticParam_6 = self.tstab

        var_shiftBoundFwd_5 = t + var_staticParam_6
        assert var_shiftBoundFwd_2 <= var_shiftBoundFwd_5
        real_prev_lb_0 = self.real_agg_lb[0]
        real_prev_ub_0 = self.real_agg_ub[0]
        real_prev_t_0 = self.real_agg_t[0]
        real_prev_v_0 = self.real_agg_v[0]
        real_cur_v_0 = -np.inf
        real_cur_t_0 = -np.inf
        self.real_agg_c[0] += 1

        if var_shiftBoundFwd_2 <=  real_prev_t_0 and\
            real_prev_t_0 <= var_shiftBoundFwd_5 and\
            real_prev_lb_0 <= var_shiftBoundFwd_2 and\
            real_prev_ub_0 <= var_shiftBoundFwd_5:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[0] += 1
            real_cur_v_0 = real_prev_v_0
            real_cur_t_0 = real_prev_t_0
            ts = self.xsrange(self.time_signal, real_prev_ub_0, var_shiftBoundFwd_5)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp >= real_cur_v_0:
                    real_cur_v_0 = tmp
                    real_cur_t_0 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_0 = -np.inf
            real_cur_t_0 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_2, var_shiftBoundFwd_5)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp >= real_cur_v_0:
                    real_cur_v_0 = tmp
                    real_cur_t_0 = tprime


        # save checkpoint
        self.real_agg_lb[0] = var_shiftBoundFwd_2
        self.real_agg_ub[0] = var_shiftBoundFwd_5
        self.real_agg_v[0] = real_cur_v_0
        self.real_agg_t[0] = real_cur_t_0
        # write result to target variable
        var_realAgg_7 = real_cur_v_0
        var_shiftBoundFwd_10 = t + var_realBinop_3
        var_shiftBoundFwd_11 = t + var_staticParam_6
        assert var_shiftBoundFwd_10 <= var_shiftBoundFwd_11
        real_prev_lb_1 = self.real_agg_lb[1]
        real_prev_ub_1 = self.real_agg_ub[1]
        real_prev_t_1 = self.real_agg_t[1]
        real_prev_v_1 = self.real_agg_v[1]
        real_cur_v_1 = np.inf
        real_cur_t_1 = -np.inf
        self.real_agg_c[1] += 1

        if var_shiftBoundFwd_10 <=  real_prev_t_1 and\
            real_prev_t_1 <= var_shiftBoundFwd_11 and\
            real_prev_lb_1 <= var_shiftBoundFwd_10 and\
            real_prev_ub_1 <= var_shiftBoundFwd_11:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[1] += 1
            real_cur_v_1 = real_prev_v_1
            real_cur_t_1 = real_prev_t_1
            ts = self.xsrange(self.time_signal, real_prev_ub_1, var_shiftBoundFwd_11)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp <= real_cur_v_1:
                    real_cur_v_1 = tmp
                    real_cur_t_1 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_1 = np.inf
            real_cur_t_1 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_10, var_shiftBoundFwd_11)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp <= real_cur_v_1:
                    real_cur_v_1 = tmp
                    real_cur_t_1 = tprime


        # save checkpoint
        self.real_agg_lb[1] = var_shiftBoundFwd_10
        self.real_agg_ub[1] = var_shiftBoundFwd_11
        self.real_agg_v[1] = real_cur_v_1
        self.real_agg_t[1] = real_cur_t_1
        # write result to target variable
        var_realAgg_12 = real_cur_v_1
        var_realBinop_1 = var_realAgg_7 - var_realAgg_12
        var_staticParam_13 = self.q_diam

        var_lt_14 = var_realBinop_1 < var_staticParam_13
        var_not_15 = not var_lt_14
        var_shiftBoundFwd_17 = t + 0.0
        var_shiftBoundFwd_18 = t + var_staticParam_6
        assert var_shiftBoundFwd_17 <= var_shiftBoundFwd_18
        real_prev_lb_2 = self.real_agg_lb[2]
        real_prev_ub_2 = self.real_agg_ub[2]
        real_prev_t_2 = self.real_agg_t[2]
        real_prev_v_2 = self.real_agg_v[2]
        real_cur_v_2 = -np.inf
        real_cur_t_2 = -np.inf
        self.real_agg_c[2] += 1

        if var_shiftBoundFwd_17 <=  real_prev_t_2 and\
            real_prev_t_2 <= var_shiftBoundFwd_18 and\
            real_prev_lb_2 <= var_shiftBoundFwd_17 and\
            real_prev_ub_2 <= var_shiftBoundFwd_18:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[2] += 1
            real_cur_v_2 = real_prev_v_2
            real_cur_t_2 = real_prev_t_2
            ts = self.xsrange(self.time_signal, real_prev_ub_2, var_shiftBoundFwd_18)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp >= real_cur_v_2:
                    real_cur_v_2 = tmp
                    real_cur_t_2 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_2 = -np.inf
            real_cur_t_2 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_17, var_shiftBoundFwd_18)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp >= real_cur_v_2:
                    real_cur_v_2 = tmp
                    real_cur_t_2 = tprime


        # save checkpoint
        self.real_agg_lb[2] = var_shiftBoundFwd_17
        self.real_agg_ub[2] = var_shiftBoundFwd_18
        self.real_agg_v[2] = real_cur_v_2
        self.real_agg_t[2] = real_cur_t_2
        # write result to target variable
        var_realAgg_19 = real_cur_v_2
        var_shiftBoundFwd_20 = t + 0.0
        var_shiftBoundFwd_21 = t + var_staticParam_6
        assert var_shiftBoundFwd_20 <= var_shiftBoundFwd_21
        real_prev_lb_3 = self.real_agg_lb[3]
        real_prev_ub_3 = self.real_agg_ub[3]
        real_prev_t_3 = self.real_agg_t[3]
        real_prev_v_3 = self.real_agg_v[3]
        real_cur_v_3 = np.inf
        real_cur_t_3 = -np.inf
        self.real_agg_c[3] += 1

        if var_shiftBoundFwd_20 <=  real_prev_t_3 and\
            real_prev_t_3 <= var_shiftBoundFwd_21 and\
            real_prev_lb_3 <= var_shiftBoundFwd_20 and\
            real_prev_ub_3 <= var_shiftBoundFwd_21:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[3] += 1
            real_cur_v_3 = real_prev_v_3
            real_cur_t_3 = real_prev_t_3
            ts = self.xsrange(self.time_signal, real_prev_ub_3, var_shiftBoundFwd_21)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp <= real_cur_v_3:
                    real_cur_v_3 = tmp
                    real_cur_t_3 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_3 = np.inf
            real_cur_t_3 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_20, var_shiftBoundFwd_21)
            for tprime in ts:
                tmp = self.fun_subTerm_8(tprime)
                if tmp <= real_cur_v_3:
                    real_cur_v_3 = tmp
                    real_cur_t_3 = tprime


        # save checkpoint
        self.real_agg_lb[3] = var_shiftBoundFwd_20
        self.real_agg_ub[3] = var_shiftBoundFwd_21
        self.real_agg_v[3] = real_cur_v_3
        self.real_agg_t[3] = real_cur_t_3
        # write result to target variable
        var_realAgg_22 = real_cur_v_3
        var_realBinop_16 = var_realAgg_19 - var_realAgg_22
        var_lt_23 = var_realBinop_16 < var_staticParam_13
        var_and_24 = var_not_15 and var_lt_23
        self.q_oob_min = min(self.q_oob_min, t)
        self.q_oob_max = max(self.q_oob_max, t)
        var_signalAtTime_9 = self.real_sig_at(self.q, t)
        var_shiftBoundFwd_37 = t + var_realBinop_3
        var_shiftBoundFwd_38 = t + var_realBinop_3
        assert var_shiftBoundFwd_37 <= var_shiftBoundFwd_38
        real_prev_lb_5 = self.real_agg_lb[5]
        real_prev_ub_5 = self.real_agg_ub[5]
        real_prev_t_5 = self.real_agg_t[5]
        real_prev_v_5 = self.real_agg_v[5]
        self.real_agg_c[5] += 1

        if var_shiftBoundFwd_37 <=  real_prev_t_5 and\
            real_prev_t_5 <= var_shiftBoundFwd_38 and\
            real_prev_lb_5 <= var_shiftBoundFwd_37 and\
            real_prev_ub_5 <= var_shiftBoundFwd_38:
            # last value is in [lb, ub] and window has shifted forward return result
            self.real_agg_h[5] += 1
            real_cur_v_5 = real_prev_v_5
            real_cur_t_5 = real_prev_t_5
        else:
            # otherwise recompute whole window
            real_cur_v_5 = 0.0
            real_cur_t_5 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_37, var_shiftBoundFwd_38)
            for tprime in ts:
                tmp = self.fun_subFormula_39(tprime)
                if tmp:
                    real_cur_v_5 = self.fun_subTerm_8(tprime)
                    real_cur_t_5 = tprime
                    break

        # save checkpoint
        self.real_agg_lb[5] = var_shiftBoundFwd_37
        self.real_agg_ub[5] = var_shiftBoundFwd_38
        self.real_agg_v[5] = real_cur_v_5
        self.real_agg_t[5] = real_cur_t_5
        # write result to target variable
        var_realUntilSmp_40 = real_cur_v_5
        var_realBinop_36 = var_signalAtTime_9 - var_realUntilSmp_40
        var_ite_41 = var_realBinop_36 if var_and_24 else 0.0
        var_staticParam_26 = self.tset

        var_shiftBoundFwd_25 = t + var_staticParam_26
        var_shiftBoundFwd_27 = t + var_staticParam_6
        assert var_shiftBoundFwd_25 <= var_shiftBoundFwd_27
        real_prev_lb_4 = self.real_agg_lb[4]
        real_prev_ub_4 = self.real_agg_ub[4]
        real_prev_t_4 = self.real_agg_t[4]
        real_prev_v_4 = self.real_agg_v[4]
        real_cur_v_4 = -np.inf
        real_cur_t_4 = -np.inf
        self.real_agg_c[4] += 1

        if var_shiftBoundFwd_25 <=  real_prev_t_4 and\
            real_prev_t_4 <= var_shiftBoundFwd_27 and\
            real_prev_lb_4 <= var_shiftBoundFwd_25 and\
            real_prev_ub_4 <= var_shiftBoundFwd_27:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[4] += 1
            real_cur_v_4 = real_prev_v_4
            real_cur_t_4 = real_prev_t_4
            ts = self.xsrange(self.time_signal, real_prev_ub_4, var_shiftBoundFwd_27)
            for tprime in ts:
                tmp = self.fun_subTerm_29(tprime)
                if tmp >= real_cur_v_4:
                    real_cur_v_4 = tmp
                    real_cur_t_4 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_4 = -np.inf
            real_cur_t_4 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_25, var_shiftBoundFwd_27)
            for tprime in ts:
                tmp = self.fun_subTerm_29(tprime)
                if tmp >= real_cur_v_4:
                    real_cur_v_4 = tmp
                    real_cur_t_4 = tprime


        # save checkpoint
        self.real_agg_lb[4] = var_shiftBoundFwd_25
        self.real_agg_ub[4] = var_shiftBoundFwd_27
        self.real_agg_v[4] = real_cur_v_4
        self.real_agg_t[4] = real_cur_t_4
        # write result to target variable
        var_realAgg_28 = real_cur_v_4
        var_staticParam_34 = self.offset_pct

        var_realUnop_35 = abs(var_ite_41)
        var_realBinop_33 = var_staticParam_34 * var_realUnop_35
        var_gt_42 = var_realAgg_28 > var_realBinop_33
        var_and_43 = var_and_24 and var_gt_42
        var_ite_59 = var_realAgg_28 if var_and_43 else 0.0
        var_gt_44 = var_ite_41 > 0.0
        var_shiftBoundFwd_45 = t + 0.0
        var_shiftBoundFwd_46 = t + var_staticParam_26
        assert var_shiftBoundFwd_45 <= var_shiftBoundFwd_46
        real_prev_lb_6 = self.real_agg_lb[6]
        real_prev_ub_6 = self.real_agg_ub[6]
        real_prev_t_6 = self.real_agg_t[6]
        real_prev_v_6 = self.real_agg_v[6]
        real_cur_v_6 = -np.inf
        real_cur_t_6 = -np.inf
        self.real_agg_c[6] += 1

        if var_shiftBoundFwd_45 <=  real_prev_t_6 and\
            real_prev_t_6 <= var_shiftBoundFwd_46 and\
            real_prev_lb_6 <= var_shiftBoundFwd_45 and\
            real_prev_ub_6 <= var_shiftBoundFwd_46:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[6] += 1
            real_cur_v_6 = real_prev_v_6
            real_cur_t_6 = real_prev_t_6
            ts = self.xsrange(self.time_signal, real_prev_ub_6, var_shiftBoundFwd_46)
            for tprime in ts:
                tmp = self.fun_subTerm_48(tprime)
                if tmp >= real_cur_v_6:
                    real_cur_v_6 = tmp
                    real_cur_t_6 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_6 = -np.inf
            real_cur_t_6 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_45, var_shiftBoundFwd_46)
            for tprime in ts:
                tmp = self.fun_subTerm_48(tprime)
                if tmp >= real_cur_v_6:
                    real_cur_v_6 = tmp
                    real_cur_t_6 = tprime


        # save checkpoint
        self.real_agg_lb[6] = var_shiftBoundFwd_45
        self.real_agg_ub[6] = var_shiftBoundFwd_46
        self.real_agg_v[6] = real_cur_v_6
        self.real_agg_t[6] = real_cur_t_6
        # write result to target variable
        var_realAgg_47 = real_cur_v_6
        var_shiftBoundFwd_49 = t + 0.0
        var_shiftBoundFwd_50 = t + var_staticParam_26
        assert var_shiftBoundFwd_49 <= var_shiftBoundFwd_50
        real_prev_lb_7 = self.real_agg_lb[7]
        real_prev_ub_7 = self.real_agg_ub[7]
        real_prev_t_7 = self.real_agg_t[7]
        real_prev_v_7 = self.real_agg_v[7]
        real_cur_v_7 = -np.inf
        real_cur_t_7 = -np.inf
        self.real_agg_c[7] += 1

        if var_shiftBoundFwd_49 <=  real_prev_t_7 and\
            real_prev_t_7 <= var_shiftBoundFwd_50 and\
            real_prev_lb_7 <= var_shiftBoundFwd_49 and\
            real_prev_ub_7 <= var_shiftBoundFwd_50:
            # last agg value is in [lb, ub] and window has shifted forward compute on [ub', ub] and update agg
            self.real_agg_h[7] += 1
            real_cur_v_7 = real_prev_v_7
            real_cur_t_7 = real_prev_t_7
            ts = self.xsrange(self.time_signal, real_prev_ub_7, var_shiftBoundFwd_50)
            for tprime in ts:
                tmp = self.fun_subTerm_52(tprime)
                if tmp >= real_cur_v_7:
                    real_cur_v_7 = tmp
                    real_cur_t_7 = tprime

        else:
            # otherwise recompute whole window
            real_cur_v_7 = -np.inf
            real_cur_t_7 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_49, var_shiftBoundFwd_50)
            for tprime in ts:
                tmp = self.fun_subTerm_52(tprime)
                if tmp >= real_cur_v_7:
                    real_cur_v_7 = tmp
                    real_cur_t_7 = tprime


        # save checkpoint
        self.real_agg_lb[7] = var_shiftBoundFwd_49
        self.real_agg_ub[7] = var_shiftBoundFwd_50
        self.real_agg_v[7] = real_cur_v_7
        self.real_agg_t[7] = real_cur_t_7
        # write result to target variable
        var_realAgg_51 = real_cur_v_7
        var_ite_54 = var_realAgg_47 if var_gt_44 else var_realAgg_51
        var_staticParam_56 = self.overshoot_pct

        var_realBinop_55 = var_staticParam_56 * var_realUnop_35
        var_gt_57 = var_ite_54 > var_realBinop_55
        var_and_58 = var_and_24 and var_gt_57
        var_ite_60 = var_ite_54 if var_and_58 else 0.0

        var_realBinop_62 = 0.0 - 1.0
        var_shiftBoundFwd_64 = t + 0.0
        var_shiftBoundFwd_65 = t + var_staticParam_6
        assert var_shiftBoundFwd_64 <= var_shiftBoundFwd_65
        real_prev_lb_8 = self.real_agg_lb[8]
        real_prev_ub_8 = self.real_agg_ub[8]
        real_prev_t_8 = self.real_agg_t[8]
        real_prev_v_8 = self.real_agg_v[8]
        self.real_agg_c[8] += 1

        if var_shiftBoundFwd_64 <=  real_prev_t_8 and\
            real_prev_t_8 <= var_shiftBoundFwd_65 and\
            real_prev_lb_8 <= var_shiftBoundFwd_64 and\
            real_prev_ub_8 <= var_shiftBoundFwd_65:
            # last value is in [lb, ub] and window has shifted forward return result
            self.real_agg_h[8] += 1
            real_cur_v_8 = real_prev_v_8
            real_cur_t_8 = real_prev_t_8
        else:
            # otherwise recompute whole window
            real_cur_v_8 = var_realBinop_62
            real_cur_t_8 = -np.inf
            ts = self.xsrange(self.time_signal, var_shiftBoundFwd_64, var_shiftBoundFwd_65)
            for tprime in ts:
                tmp = self.fun_subFormula_68(tprime)
                if tmp:
                    real_cur_v_8 = self.fun_subTerm_66(tprime)
                    real_cur_t_8 = tprime
                    break

        # save checkpoint
        self.real_agg_lb[8] = var_shiftBoundFwd_64
        self.real_agg_ub[8] = var_shiftBoundFwd_65
        self.real_agg_v[8] = real_cur_v_8
        self.real_agg_t[8] = real_cur_t_8
        # write result to target variable
        var_realUntilSmp_73 = real_cur_v_8
        self.tt_oob_min = min(self.tt_oob_min, t)
        self.tt_oob_max = max(self.tt_oob_max, t)
        var_signalAtTime_67 = self.real_sig_at(self.tt, t)
        var_realBinop_63 = var_realUntilSmp_73 - var_signalAtTime_67
        var_ite_74 = var_realBinop_63 if var_and_24 else var_realBinop_62
        var_realBinopPrefix_61 = max(var_realBinop_62, var_ite_74)

        return var_and_24, var_ite_41, var_and_43, var_ite_59, var_and_58, var_ite_60, var_realBinopPrefix_61

    def fun_subFormula_68(self, t):
        self.x_oob_min = min(self.x_oob_min, t)
        self.x_oob_max = max(self.x_oob_max, t)
        var_signalAtTime_32 = self.real_sig_at(self.x, t)
        self.q_oob_min = min(self.q_oob_min, t)
        self.q_oob_max = max(self.q_oob_max, t)
        var_signalAtTime_9 = self.real_sig_at(self.q, t)
        var_realBinop_31 = var_signalAtTime_32 - var_signalAtTime_9
        var_realUnop_30 = abs(var_realBinop_31)
        var_staticParam_70 = self.time_to_reach_pct

        var_realUnop_71 = abs(var_signalAtTime_9)
        var_realBinop_69 = var_staticParam_70 * var_realUnop_71
        var_lt_72 = var_realUnop_30 < var_realBinop_69
        return var_lt_72

    def fun_subTerm_66(self, t):
        self.tt_oob_min = min(self.tt_oob_min, t)
        self.tt_oob_max = max(self.tt_oob_max, t)
        var_signalAtTime_67 = self.real_sig_at(self.tt, t)
        return var_signalAtTime_67

    def fun_subTerm_52(self, t):
        self.q_oob_min = min(self.q_oob_min, t)
        self.q_oob_max = max(self.q_oob_max, t)
        var_signalAtTime_9 = self.real_sig_at(self.q, t)
        self.x_oob_min = min(self.x_oob_min, t)
        self.x_oob_max = max(self.x_oob_max, t)
        var_signalAtTime_32 = self.real_sig_at(self.x, t)
        var_realBinop_53 = var_signalAtTime_9 - var_signalAtTime_32
        return var_realBinop_53

    def fun_subTerm_48(self, t):
        self.x_oob_min = min(self.x_oob_min, t)
        self.x_oob_max = max(self.x_oob_max, t)
        var_signalAtTime_32 = self.real_sig_at(self.x, t)
        self.q_oob_min = min(self.q_oob_min, t)
        self.q_oob_max = max(self.q_oob_max, t)
        var_signalAtTime_9 = self.real_sig_at(self.q, t)
        var_realBinop_31 = var_signalAtTime_32 - var_signalAtTime_9
        return var_realBinop_31

    def fun_subTerm_29(self, t):
        self.x_oob_min = min(self.x_oob_min, t)
        self.x_oob_max = max(self.x_oob_max, t)
        var_signalAtTime_32 = self.real_sig_at(self.x, t)
        self.q_oob_min = min(self.q_oob_min, t)
        self.q_oob_max = max(self.q_oob_max, t)
        var_signalAtTime_9 = self.real_sig_at(self.q, t)
        var_realBinop_31 = var_signalAtTime_32 - var_signalAtTime_9
        var_realUnop_30 = abs(var_realBinop_31)
        return var_realUnop_30

    def fun_subFormula_39(self, t):
        return True

    def fun_subTerm_8(self, t):
        self.q_oob_min = min(self.q_oob_min, t)
        self.q_oob_max = max(self.q_oob_max, t)
        var_signalAtTime_9 = self.real_sig_at(self.q, t)
        return var_signalAtTime_9

    def oob(self):
        """
        True if some signal was accessed out of its bounds.
        """
        oob_min = self.tt_oob_min < self.time_signal[0] or self.x_oob_min < self.time_signal[0] or self.q_oob_min < self.time_signal[0]
        oob_max = self.tt_oob_max > self.time_signal[-1] or self.x_oob_max > self.time_signal[-1] or self.q_oob_max > self.time_signal[-1]
        return oob_min or oob_max


    def signum(self, x):
        """
        Utility method, not built-in in python.
        """
        if x==0.0:
            return 0.0
        elif x > 0.0:
            return 1.0
        else:
            return -1.0

    def xsrange(self, xs, x_min, x_max):
        """
         Returns generator
             [xs[i], ..., xs[j]]
         where:
         - xs[i-1] <  x_min <= xs[i]
         -   xs[j] <= x_max < xs[j+1]
         :param x_min: num
         :param x_max: num
         :return: generator
         """
        assert x_min <= x_max
        i = np.searchsorted(xs, x_min, side='left')
        j = np.searchsorted(xs, x_max, side='right')
        res = [xs[i] for i in range(i, j)]
        if len(res) == 0:
            return [x_min, x_max]
        if res[0] > x_min:
            res.insert(0, x_min)
        if res[-1] < x_max:
            res.append(x_max)
        return res

    def bool_sig_at(self, x, t):
        i = np.searchsorted(self.time_signal, t, side='right') - 1
        if i == -1:
            return False
        else:
            return x[i]

    def real_sig_at(self, x, t):
        i = np.searchsorted(self.time_signal, t, side='right') - 1
        if i == -1:
            return -np.inf
        else:
            return x[i]

class AllPropsComp:
    """
    Companion object for the class AllProps.
    Hase methods that cannot be jitted using Numba.
    """
    @staticmethod
    def eval_all(obs):
        """
        Evaluates the property for all timesteps (for use in jupyter notebook)
        """
        obs.last_t = -np.inf
        res = np.array([obs.apply(each) for each in obs.time_signal])
        return res

    @staticmethod
    def eval_and_plot(obs):
        """
        shows a plot of the signals and property for all timesteps (for use in jupyter notebook)
        """
        res = AllPropsComp.eval_all(obs)
        plt.plot(obs.time_signal, res)
        plt.plot(obs.time_signal, obs.tt)
        plt.plot(obs.time_signal, obs.x)
        plt.plot(obs.time_signal, obs.q)
        plt.show()


    @staticmethod
    def print_cache_stats(obs):

        for i in range(len(obs.real_agg_h)):
            h = obs.real_agg_h[i]
            c = obs.real_agg_c[i]
            print(f'real_agg hits: {h}/{c}  = {100.0 * h/c}%')

        for i in range(len(obs.bool_agg_h)):
            h = obs.bool_agg_h[i]
            c = obs.bool_agg_c[i]
            print(f'bool_agg hits: {h}/{c}  = {100.0 * h/c}%')

