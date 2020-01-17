import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def spider_plot(title, radial_labels, case_data, case_labels):
    """
    Minimum example of the inputs
    :param title: 'Titulo'
    :param radial_labels: ["medida 1", "medida 2", "medida 3"]
    :param case_data: [[0.1, 0.2, 0.5],
                       [0.3, 0.4, 0.6]]
    :param case_labels: ["Serie 1", "Serie 2"]
    :return:
    """
    n_radial_labels = len(radial_labels)

    # What will be the angle of each axis in the plot?
    angles = [np.pi * (0.5 + (1 + 2 * n) / float(n_radial_labels)) for n in range(n_radial_labels)]
    for i, a in enumerate(angles):
        if a > np.pi*2:
            angles[i] -= np.pi*2
    angles += angles[:1]

    # Initialise the spider plot
    fig = plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True, frame_on=False)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], radial_labels, size='large')

    ax.plot(angles, [1.0]*len(angles), linewidth=1, linestyle='solid', color='black')

    # grey half circle
    half_axis = np.concatenate(([angles[0] - np.pi / 14], angles[:8], [angles[7] + np.pi / 14] * 2))
    half_circle = [0.98] + [1] * 8 + [0.98] + [0]
    ax.fill(half_axis, half_circle, facecolor="grey", alpha=0.25)  # Fill the polygon

    #    axes.set_title(title, weight='bold', position=(0.5, 1.1),
    #                   horizontalalignment='center', verticalalignment='center')

    colors = ["C%d" % i for i in range(len(case_labels))]
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2])
    ax.set_ylim([0, 1])

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    for i, (cl, color) in enumerate(zip(case_labels, colors)):
        values = case_data[i]  # df.loc[1].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cl, color=color)

    # Block- and frame-based legend
    frame_patch = mpatches.Patch(facecolor='lightgrey', edgecolor='k', label='Frame-based metrics')
    block_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Block-based metrics')
    first_legend = plt.legend(handles=[frame_patch, block_patch], ncol=1,
                              bbox_to_anchor=(-0.15, 0.0), loc=2, borderaxespad=0.0,
                              framealpha=0, fontsize='large')
    plt.gca().add_artist(first_legend)

    # Add legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05),
               labelspacing=0.5, fontsize='large')
    # plt.legend(case_labels, bbox_to_anchor=(1.15, 1.05),  # loc=(0.9, .95),
    #            loc=2, labelspacing=0.5, fontsize='small')

    plt.tight_layout(pad=3.5)
    plt.savefig(title + ".pdf")
    plt.savefig(title + ".png")
    plt.show()


def single_spider_df_summaries(summaries, labels, title="Titulo"):
    case_data = []
    for summary, lab in zip(summaries, labels):
        summary_mean = summary.mean()
        # Saving data to plot
        case_data.append([summary_mean.frame_recall, summary_mean.frame_precision,
                          1 - summary_mean.f_rate, 1 - summary_mean.m_rate,
                          1 - summary_mean.d_rate, 1 - summary_mean.i_rate,
                          1 - summary_mean.u_rate, 1 - summary_mean.o_rate,
                          #
                          1 - summary_mean.ins_rate, 1 - summary_mean.del_rate,
                          1 - summary_mean.merge_rate, 1 - summary_mean.frag_rate,
                          summary_mean.event_precision, summary_mean.event_recall
                          ])

    spider_plot(title=title,
                # Labels are organized
                radial_labels=[  # Labels are given in anti-clockwise direction
                    # Frame-based metrics
                    # "FNR", "FDR", "Frag.", "Merg.", "Del.", "Ins.",
                    r"$\rm{FNR}_f$", r"$\rm{FDR}_f$", r"$\rm{F}_f$", r"$\rm{M}_f$", r"$\rm{D}_f$", r"$\rm{I}_f$",
                    # "Under.", "Over.",
                    r"$\rm{U}_f$", r"$\rm{O}_f$",
                    # Block-based metrics
                    # "Ins.", "Del.", "Merg.", "Frag.", "FDR", "FNR"
                    r"$\rm{I}_b$", r"$\rm{D}_b$", r"$\rm{M}_b$", r"$\rm{F}_b$", r"$\rm{FDR}_b$", r"$\rm{FNR}_b$"
                ],
                case_data=case_data,
                case_labels=labels)


def spider_df_summaries(summaries_by_activity, labels):
    for act in summaries_by_activity[0].mean().index.tolist():
        single_spider_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                   labels, act)
        print_f1scores_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                    labels, act)


def print_f1scores_df_summaries(summaries, labels, act):
    print("Label".ljust(25) + "Frame-based f1score".ljust(25) + "Block-based f1score")
    for summary, lab in zip(summaries, labels):
        frame_str = "%0.3f (+-%0.3f)" % (summary.frame_f1score.mean(), summary.frame_f1score.std())
        event_str = "%0.3f (+-%0.3f)" % (summary.event_f1score.mean(), summary.event_f1score.std())
        print(lab.ljust(25)[:25] + frame_str.ljust(25) + event_str)


def spider_and_violinplot_df_summaries(summaries_by_activity, labels):

    activities_of_interests = summaries_by_activity[0].groups.keys()

    for activity in activities_of_interests:
        print('\n', activity, '\n')
        summaries = [s.get_group(activity) for s in summaries_by_activity]

        to_save_dic = {}
        for summary, lab in zip(summaries, labels):
            minutes_errors = summary.raw_time_error.values / 60.0
            minutes_positives = summary.ground_positives / 60.0
            to_save_dic["filename"] = summary.ground_filename
            to_save_dic[lab + "_diff_minutes"] = minutes_errors
            to_save_dic["ground_minutes"] = minutes_positives
        df = pd.DataFrame(to_save_dic)
        df.to_csv("time_errors_" + activity, index=False)

        mean_duration = df["ground_minutes"].mean()

        single_spider_df_summaries(summaries, labels, activity)
        print_f1scores_df_summaries(summaries, labels, activity)
        violinplot_raw_errors_df_summaries(summaries, labels, activity, mean_duration)

        # saving frame scores TODO maybe this should be a function
        df_frame_fscores = pd.DataFrame()
        df_block_fscores = pd.DataFrame()
        for summary, lab in zip(summaries, labels):
            df_frame_fscores[lab] = summary.frame_f1score
            df_block_fscores[lab] = summary.event_f1score
        df_frame_fscores.to_csv("fscores_frame_" + activity, index=False)
        df_block_fscores.to_csv("fscores_block_" + activity, index=False)


def violinplot_raw_errors_df_summaries(summaries, labels, act, mean_duration):
    plt.figure()
    pos = np.arange(len(labels)) + .5  # the bar centers on the y axis

    if len(labels) > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    colors = ["C%d" % i for i in range(len(labels))]

    to_print = []
    for summary, lab, p in zip(summaries, labels, pos):
        minute_errors = summary.raw_time_error.values / 60.0
        to_print.append(" ".join(["%.2f" % i for i in minute_errors]) +
                        "\t(%.2f)_mean\t" % np.mean(minute_errors) +
                        "\t(%.2f)_median\t" % np.median(minute_errors) + lab)
        plt.violinplot(minute_errors[np.isfinite(minute_errors)], [p], points=50, vert=False, widths=0.65,
                       showmeans=False, showmedians=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.yticks(pos, labels)
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.xlabel('Time Estimation Error (min)')

    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(["%.0f%%" % (100.0 * d / mean_duration) for d in ax1.get_xticks()])
    ax2.set_xlabel("Normalized Error")
    plt.minorticks_on()

    plt.tight_layout()

    plt.savefig('violin' + act + '.pdf')
    plt.savefig('violin' + act + '.png')

    plt.show()

    print("Mean duration used to normalize: %.2f" % mean_duration)
    print("Time estimation error per signal")
    for row in to_print:
        print(row)
