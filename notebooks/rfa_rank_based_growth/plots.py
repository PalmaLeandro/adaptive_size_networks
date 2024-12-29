"""Implements plots for the RFA-based growth experiment."""

import os, sys, numpy, functools, matplotlib.pyplot

sys.path.append(os.path.abspath(os.path.join('..')))  # Allow repository modules to be imported

from utils.plotting import draw_figure_into_canvas, plot_samples_and_model_activation_on_canvas

def plot_effective_rank_and_detected_noise(run=None, sample_size=None, batch_size=None, noise_rate=None, epoch=None, canvas=None, **kwargs):     
    fig, (rs_ax, rs_ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(12, 6))
    rs_ax.set_title('Effective rank (r)', fontname='Times New Roman')
    rs_ax.set_xlabel('iterations', fontname='Times New Roman')
    rs_ax.set_ylabel('r_k', fontname='Times New Roman')
    rs_ax2.set_title('Effective rank (r) label', fontname='Times New Roman')
    rs_ax2.set_xlabel('iterations', fontname='Times New Roman')
    rs_ax2.set_ylabel('r_k', fontname='Times New Roman')

    for ax in (rs_ax, rs_ax2): 
        for tick in ax.get_xticklabels(): tick.set_fontname('Times New Roman')
        for tick in ax.get_yticklabels(): tick.set_fontname('Times New Roman')

    iterations = [iteration * sample_size / batch_size for iteration in range(1, epoch + 1)]
    for k, (activations_rs1, activations_rs2) in enumerate(zip(numpy.array(run['activations_rs1']).transpose(), 
                                                               numpy.array(run['activations_rs2']).transpose())):
       rs_ax.plot(iterations, activations_rs1, label=str(k))
       rs_ax2.plot(iterations, activations_rs2, label=str(k))
    
    rs_ax.legend(title='k', prop=dict(family='Times New Roman'), title_fontproperties=dict(family='Times New Roman'))
    rs_ax2.legend(title='k', prop=dict(family='Times New Roman'), title_fontproperties=dict(family='Times New Roman'))
    if canvas is not None: draw_figure_into_canvas(fig, canvas)

def plot_effective_rank_and_detected_noise_on_canvas(canvas, **kwargs):
   return functools.partial(plot_effective_rank_and_detected_noise, canvas=canvas)