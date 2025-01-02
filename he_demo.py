from he import *

import ipywidgets as widgets
import matplotlib.pyplot as plt

class HeDemo:

    BACKGROUND_COLOR="#242322"

    DEBUG=0

    def __init__(self):
        self.he = HarmonicEntropy(N=100, a=7, limit=1200, s=17, verbose=self.DEBUG)
        self.he3 = HarmonicEntropy(N=100, a=7, limit=1200, s=17, he3=True, verbose=self.DEBUG)

        self.figure = None
        self.entropy1 = None
        self.entropy2 = None

        self.default_limit=2

        self.last_update = {}

        self.limit_option_harmonics = [2,3,4,5,6,7,8]
        self.limit_option_cents = [ ratio_to_cents(h) for h in self.limit_option_harmonics ]
        self.limit_options = [ (str(self.limit_option_harmonics[i]), self.limit_option_cents[i]) for i in range(len(self.limit_option_harmonics)) ]

        self.n_size_select = widgets.Dropdown(options=[10,25,50,100,200,500], value=100, description="N")
        self.limit_select = widgets.Dropdown(options=self.limit_options, value=1200.0, description="Harmonic Limit")
        self.alpha_slider = widgets.FloatSlider(min=1e-4, max=10, value=7, step=0.01, description="alpha")
        self.spread_slider = widgets.FloatSlider(min=1, max=25, value=12, step=1, description="spread")
        self.weight_select = widgets.Dropdown(options=["default", "lencf", "lenmaxcf", "sumcf", "custom"], value="default", description="Weighting")
        self.scroll_widget = widgets.FloatRangeSlider(min=0, max=ratio_to_cents(self.default_limit), step=1, value=(0, 1200), description="Scroll")

        # for now, these expect strings (cmdline reasons)
        self.prime_options = [ str(p) for p in [ 1, *PRIMES[:12] ] ]
        self.prime_limit_select = widgets.Dropdown(options=[None, *self.prime_options], value=None, description="Prime Limit")

        self.basis_filter_options = ["default", *HarmonicEntropy.BASIS_MASK_OPTIONS]
        self.basis_filter_select = widgets.Dropdown(options=self.basis_filter_options, description="Filter Options", value=None)

        self.transform_options = ["default", *HarmonicEntropy.BASIS_TRANSFORM_OPTIONS]
        self.transform_select = widgets.Dropdown(options=self.transform_options, description="Basis Transform", value=None)

        self.ypad_slider = widgets.FloatSlider(min=-10, max=10, value=0, description='y-adjust')

        self.save_button = widgets.Button(description="Save")

    def update(self, **kwArgs):
        scroll = None
        if kwArgs['scroll'] is not None:
            scroll = kwArgs['scroll']
            del kwArgs['scroll']

        update_args = { k:v for k,v in kwArgs.items() if v is not None and (k not in self.last_update or v != self.last_update[k])}

        if kwArgs['bfilter'] is not None:
            update_args[kwArgs['bfilter']] = True
            del update_args['bfilter']

        if self.DEBUG:
            print(f'last update: {self.last_update}')
            print(f'demo update: {update_args}')

        self.he.update(update_args)
        self.he3.update(update_args)

        for k in update_args:
            self.last_update[k] = update_args[k]

        self.he.calculate()
        self.he3.calculate()

        self.figure, [ self.entropy1,
                       self.entropy2 ] = plt.subplots(ncols=2, figsize=(15, 5), gridspec_kw={'width_ratios':[5,3], 'height_ratios':[1]})
        # self.figure.set_facecolor(self.BACKGROUND_COLOR)

        (x1,y1) = self.he.getEntropyPlotData(scroll[0], scroll[1])
        self.entropy1.plot(x1, y1)
        y_max = y1.max()
        if y_max < 6:
            y_max = 6
        self.entropy1.set_ylim((0, y_max))

        (xy2_grid, z) = self.he3.getEntropyPlotData(scroll[0], scroll[1])
        self.entropy2.imshow(z, cmap="inferno", aspect="equal", origin="lower")
        self.entropy2.set_axis_off()

    def update_scroll_max(self, *args):
        cents = ratio_to_cents(args[0].new)
        lastMax = self.scroll_widget.max
        if cents != lastMax:
            self.scroll_widget.max = cents
            self.scroll_widget.value = (self.scroll_widget.min, self.scroll_widget.max)
    

    def createControls(self):
        self.limit_select.observe(self.update_scroll_max, 'value')
        self.save_button.on_click = lambda b: self.he3.makePlot(True, False)

        modelBox = widgets.VBox([
            self.n_size_select,
            self.limit_select,
            self.alpha_slider,
            self.weight_select,
            self.spread_slider,
            self.scroll_widget
            ])
        # modelBox.add_class('box_style')
        
        basisControls = widgets.VBox([
            self.prime_limit_select,
            self.basis_filter_select,
            self.transform_select,
            ])
        # basisControls.add_class('box_style')
        
        debugControls = widgets.VBox([
            self.ypad_slider,
            self.save_button
        ])
        # debugControls.add_class('box_style')


        ui = widgets.HBox([modelBox, basisControls, debugControls])
        # ui.add_class('box_style')

        out = widgets.interactive_output(self.update, {
                    'N'       : self.n_size_select,
                    'limit'   : self.limit_select,
                    'weight'  : self.weight_select,
                    's'       : self.spread_slider,
                    'a'       : self.alpha_slider,
                    'scroll'  : self.scroll_widget,
                    'tx'      : self.transform_select,
                    'p_limit' : self.prime_limit_select,
                    'bfilter' : self.basis_filter_select,
                    'ypad'    : self.ypad_slider,
                    })
        # out.add_class('box_style')
        
        return (ui, out)
