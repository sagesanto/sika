from sika.implementations.spectroscopy.kpic.kpic_dataloader import KPICDataLoader
from sika.implementations.spectroscopy.kpic.kpic_spectrum import KPICSpectrum
from sika.implementations.spectroscopy.spectra.spectrum import Spectrum
from sika.modeling import Dataset, Model
from sika.modeling.parameter_set import EmptyParameterSet

__all__ = ["EmpiricalKPICModel"]

class EmpiricalKPICModel(Model[Spectrum]):
    """
    A model for a companion star that uses an empirical CRIRES spectrum.
    """

    def __init__(self, target_name, *args, normalize=True, **kwargs):
        super().__init__(EmptyParameterSet(), *args, **kwargs)
        self.empirical_spectrum: Dataset[KPICSpectrum] = None
        self.target_name = target_name
        self.disp_name = None
        self.ds = None
        self.normalize=normalize
        self.loader = KPICDataLoader(normalize=self.normalize)

    @property
    def previous(self):
        return [self.loader]

    def args_to_dict(self):
        return {"target_name": self.target_name}

    def _setup(self):
        self.loader.configure(self.config, self.logger)
        self.empirical_spectrum = self.loader({"target": self.target_name})
        self.disp_name = self.config.get(self.target_name, {}).get("display_name", None)

    def make_model(self, *args, **kwargs) -> Dataset[KPICSpectrum]:
        """
        Generate a model spectrum for the companion using the empirical CRIRES data.
        """
        return self.empirical_spectrum

    @property
    def display_name(self):
        if self.disp_name is not None:
            return self.disp_name
        return self.target_name