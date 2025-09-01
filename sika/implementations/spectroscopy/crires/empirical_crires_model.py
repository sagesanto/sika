from sika.implementations.spectroscopy.crires.crires_loader import CRIRESDataLoader
from sika.implementations.spectroscopy.crires.crires_spectrum import CRIRESSpectrum
from sika.implementations.spectroscopy.spectra.spectrum import Spectrum
from sika.modeling import Dataset, Model
from sika.modeling.parameter_set import EmptyParameterSet

__all__ = ["EmpiricalCRIRESModel"]

class EmpiricalCRIRESModel(Model[Spectrum]):
    """
    A model for a companion star that uses an empirical CRIRES spectrum.
    """

    def __init__(self, target_name, *args, **kwargs):
        super().__init__(EmptyParameterSet(), *args, **kwargs)
        self.empirical_spectrum: Dataset[CRIRESSpectrum] = None
        self.target_name = target_name
        self.disp_name = None
        self.ds = None
        self.loader = CRIRESDataLoader()

    @property
    def previous(self):
        return [self.loader]

    def args_to_dict(self):
        return {"target_name": self.target_name}

    def _setup(self):
        self.loader.configure(self.config, self.logger)
        self.empirical_spectrum = self.loader({"target": self.target_name})
        self.disp_name = self.config.get(self.target_name, {}).get("display_name", None)

    def make_model(self, *args, **kwargs) -> Dataset[CRIRESSpectrum]:
        """
        Generate a model spectrum for the companion using the empirical CRIRES data.
        """
        return self.empirical_spectrum

    @property
    def display_name(self):
        if self.disp_name is not None:
            return self.disp_name
        return self.target_name