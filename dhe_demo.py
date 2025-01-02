from dhe import *

import ipywidgets as widgets
import matplotlib.pyplot as plt

class DheDemo:

    BACKGROUND_COLOR="#242322"

    DEBUG=0

    def __init__(self):

        self.dhe = DeltaHarmonicEntropy()

        self.figure = None
        self.plot_he2 = None
        self.plot_dhe = None
        self.plot_combined = None
        self.axes = None

        self.default_limit = 2

        self.last_update = {}

        self.limit_option_harmonics = [2,3,4,5,6,7,8]
        # self.limit_option_cents = [ ratio_to_cents(h) for h in self.limit_option_harmonics ]
        # self.limit_options = [ (str(self.limit_option_harmonics[i]), self.limit_option_cents[i]) for i in range(len(self.limit_option_harmonics)) ]
        self.limit_options = [ (str(harmonic), harmonic) for harmonic in self.limit_option_harmonics ]

        self.n_size_select = widgets.Dropdown(options=[10,25,50,100,200,500], value=100, description="Int Limit")
        self.limit_select = widgets.Dropdown(options=self.limit_options, value=2, description="Range")
        self.alpha_slider = widgets.FloatSlider(min=1e-4, max=10, value=7, step=0.01, description="alpha")
        self.spread_slider = widgets.FloatSlider(min=1, max=25, value=12, step=1, description="spread")
        # self.weight_select = widgets.Dropdown(options=["default", "lencf", "lenmaxcf", "sumcf", "custom"], value="default", description="Weighting")
        # self.scroll_widget = widgets.FloatRangeSlider(min=0, max=ratio_to_cents(self.default_limit), step=1, value=(0, 1200), description="Scroll")

        # self.ypad_slider = widgets.FloatSlider(min=-10, max=10, value=0, description='y-adjust')

        self.save_button = widgets.Button(description="Save")

    def update(self, **kwArgs):
        # scroll = None
        # if kwArgs['scroll'] is not None:
        #     scroll = kwArgs['scroll']
        #     del kwArgs['scroll']

        update_args = { k:v for k,v in kwArgs.items() if v is not None and (k not in self.last_update or v != self.last_update[k])}

        # if kwArgs['bfilter'] is not None:
        #     update_args[kwArgs['bfilter']] = True
        #     del update_args['bfilter']

        if self.DEBUG:
            print(f'last update: {self.last_update}')
            print(f'demo update: {update_args}')

        self.dhe.update(**update_args)

        for k in update_args:
            self.last_update[k] = update_args[k]

        self.dhe.calculate()

        self.figure, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.figure.suptitle('2HE with Difference-Tone 3HE')

        x_axis = self.dhe.x_cents
        # y_max = max(self.dhe.he2.Entropy.max(), self.dhe.differenceEntropy.max(), self.dhe.combinedEntropy.max())
        y_max = 7

        self.axes[0].plot(x_axis, self.dhe.he2.Entropy, label="2HE")
        self.axes[0].set_ylim((0, y_max))
        self.axes[0].set_ylabel("2HE")
        self.axes[1].plot(x_axis, self.dhe.differenceEntropy, label="DHE", color='g')
        self.axes[1].set_ylim((0, y_max))
        self.axes[1].set_ylabel("DHE")
        self.axes[2].plot(x_axis, self.dhe.combinedEntropy, label="D2HE", color='orange')
        self.axes[2].set_ylim((0, y_max))
        self.axes[2].set_ylabel("D2HE")

    # def update_scroll_max(self, *args):
    #     cents = ratio_to_cents(args[0].new)
    #     lastMax = self.scroll_widget.max
    #     if cents != lastMax:
    #         self.scroll_widget.max = cents
    #         self.scroll_widget.value = (self.scroll_widget.min, self.scroll_widget.max)
    

    def createControls(self):
        # self.limit_select.observe(self.update_scroll_max, 'value')
        # self.save_button.on_click = lambda b: self.he3.makePlot(True, False)

        modelBox = widgets.VBox([
            self.n_size_select,
            self.limit_select,
            self.alpha_slider,
            # self.weight_select,
            self.spread_slider,
            # self.scroll_widget
            ])
        # modelBox.add_class('box_style')
        
        # basisControls = widgets.VBox([
        #     self.prime_limit_select,
        #     self.basis_filter_select,
        #     self.transform_select,
        #     ])
        # basisControls.add_class('box_style')
        
        # debugControls = widgets.VBox([
        #     self.ypad_slider,
        #     self.save_button
        # ])
        # debugControls.add_class('box_style')


        # ui = widgets.HBox([modelBox, basisControls, debugControls])
        ui = widgets.HBox([modelBox])
        # ui.add_class('box_style')

        out = widgets.interactive_output(self.update, {
                    'int_limit' : self.n_size_select,
                    'x_range'   : self.limit_select,
                    # 'weight'  : self.weight_select,
                    'spread'    : self.spread_slider,
                    'alpha'     : self.alpha_slider,
                    # 'scroll'  : self.scroll_widget,
                    # 'tx'      : self.transform_select,
                    # 'p_limit' : self.prime_limit_select,
                    # 'bfilter' : self.basis_filter_select,
                    # 'ypad'    : self.ypad_slider,
                    })
        # out.add_class('box_style')
        
        return (ui, out)

if __name__ == '__main__':
    demo = DheDemo()
    demo.update(int_limt=100, x_range=4, alpha=4, spread=10)
    demo.figure.show()
    