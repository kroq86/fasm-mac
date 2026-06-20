#pragma once

namespace eml_core {

extern "C" double lb_eml_f64(double x, double y);
extern "C" double lb_eml_exp_f64(double x);
extern "C" double lb_eml_log_f64(double x);
extern "C" double lb_eml_sub_f64(double a, double b);
extern "C" double lb_eml_neg_f64(double x);
extern "C" double lb_eml_add_f64(double a, double b);
extern "C" double lb_eml_inv_f64(double x);
extern "C" double lb_eml_mul_f64(double a, double b);
extern "C" double lb_eml_div_f64(double a, double b);
extern "C" double lb_eml_half_f64(double x);

}  // namespace eml_core
