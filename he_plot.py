import matplotlib.pyplot as plt

def create_plot(data, title="Plot", xlabel="X", ylabel="Y", **kwArgs):
    default_label = ylabel
    if "label-default" in kwArgs:
        default_label = kwArgs["label-default"]
        del kwArgs["label-default"]
    if "imshow" in kwArgs:
        cmap = None if "cmap" not in kwArgs else kwArgs["cmap"]
        if "figsize" in kwArgs:
            plt.figure(figsize=kwArgs["figsize"])
        origin = "lower" if "origin" not in kwArgs else kwArgs["origin"]
        plt.imshow(data, cmap=cmap, aspect="equal", origin=origin)
        plt.axis('off')
    else:
        plt.plot(data, label=default_label)
        
        num_plots=1

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if kwArgs:
            for kw in kwArgs:
                if kw == "annotate":
                    txtpos = (0, max(data))
                    plt.annotate(kwArgs[kw], txtpos)

                elif kw == "xticks":
                    labels=None
                    if "xtickLabels" in kwArgs:
                        labels = kwArgs["xtickLabels"]
                    plt.xticks(kwArgs[kw], labels=labels)

                elif kw.startswith("plot"):
                    if kw.endswith("label"):
                        continue
                    
                    labelKw = f'{kw}-label'
                    label = kwArgs[labelKw]
                    plt.plot(kwArgs[kw], label=label)
                    num_plots += 1

        if num_plots > 1:
            plt.legend()
            # primary_plot.set_label(kwArgs['label-default'])

    return plt
