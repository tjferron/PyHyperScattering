from PyHyperScattering.FileLoader import FileLoader
import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings
import re
import PyHyperScattering

try:
    from astropy.io import fits
except ImportError:
    warnings.warn(
        "Could not import astropy.io.fits, needed for ALS 11.0.1.2 RSoXS loading.  Is this dependency installed?",
        stacklevel=2,
    )


class ALS11012RSoXSLoader(FileLoader):
    """
    Loader for FITS files from the ALS 11.0.1.2 RSoXS instrument


    Additional requirement: astropy, for FITS file loader


    Usage is mainly via the inherited function integrateImageStack from FileLoader

    """

    file_ext = "(.*?).fits"
    md_loading_is_quick = True

    def __init__(
        self,
        corr_mode=None,
        user_corr_func=None,
        dark_pedestal=0,
        exposure_offset=0.000,
        dark_subtract=False,
        data_collected_after_mar2021=False,
        constant_md={},
    ):
        """
        Args:
            corr_mode (str): origin to use for the intensity correction.  Can be 'expt','i0','expt+i0','user_func','old',or 'none'
            user_corr_func (callable): that takes the header dictionary and returns the value of the correction.
            dark_pedestal (numeric): number to add to the whole image before doing dark subtraction, to avoid non-negative values.
            exposure_offset (numeric): value to add to the exposure time.  Measured at 2ms with the piezo shutter in Dec 2019 by Jacob Thelen, NIST
            data_collected_after_mar2021 (boolean, default False): if True, uses 'CCD Camera Shutter Inhibit' as the dark-indicator; if False, uses 'CCD Shutter Inhibit'
            constant_md (dict): values to insert into every metadata load.  Example: beamcenter_x, beamcenter_y, sdd to enable qx/qy loading.
        """
        if corr_mode == None:
            warnings.warn(
                "Correction mode was not set, not performing *any* intensity corrections.  Are you sure this is "
                + "right? Set corr_mode to 'none' to suppress this warning.",
                stacklevel=2,
            )
            self.corr_mode = "none"
        else:
            self.corr_mode = corr_mode

        if data_collected_after_mar2021 is None:
            warnings.warn(
                "The default behavior will change in PyHyperScattering 0.3 to assume data was collected after March 2021.  Set kwarg explicitly to override.",
                DeprecationWarning,
            )
            if PyHyperScattering.__version__ < 0.3:
                data_collected_after_mar2021 = False
            else:
                data_collected_after_mar2021 = True

        if data_collected_after_mar2021:
            self.shutter_inhibit = "CCD Camera Shutter Inhibit"
        else:
            self.shutter_inhibit = "CCD Shutter Inhibit"
        self.dark_pedestal = dark_pedestal
        self.user_corr_func = user_corr_func
        self.exposure_offset = exposure_offset
        self.darks = {}
        self.constant_md = constant_md
        self.dark_subtract = dark_subtract

        # ALS normalization
        self.i0 = None
        self.i1 = None

    def loadDarks(self, basepath, dark_base_name):
        """
        Load a series of dark images as a function of exposure time, to be subtracted from subsequently-loaded data.

        Args:
            basepath (str or Path): path to load images from
            dark_base_name (str): str that must be in file for file to be a dark
        """
        for file in os.listdir(basepath):
            if dark_base_name in file:
                darkimage = fits.open(basepath + file)
                assert darkimage[0].header[self.shutter_inhibit] == 1, (
                    "CCD Shutter was not inhibited for image "
                    + file
                    + "... probably not a dark."
                )

                exptime = round(darkimage[0].header["EXPOSURE"], 2)

                self.darks[exptime] = darkimage[2].data

    def loadSampleSpecificDarks(
        self, basepath, file_filter="", file_skip="donotskip", md_filter={}
    ):
        """
        load darks matching a specific sample metadata

        Used, e.g., to load darks taken at a time of measurement in order to improve the connection between the dark and sample data.

        Args:
            basepath (str): path to load darks from
            file_filter (str): string that must be in each file name
            file_skip (str): string that, if in file name, means file should be skipped.
            md_filter (dict): dict of required metadata values.  this will be appended with dark images only, no need to put that here.
        """

        md_filter.update({self.shutter_inhibit: 1})

        for file in os.listdir(basepath):
            if (
                (re.match(self.file_ext, file) is not None)
                and file_filter in file
                and file_skip not in file
            ):
                if self.md_loading_is_quick:
                    # if metadata loading is quick, we can just peek at the metadata and decide what to do
                    md = self.peekAtMd(basepath + file)
                    img = None
                else:
                    input_image = fits.open(basepath + file)
                    md = self.normalizeMetadata(
                        dict(
                            zip(
                                input_image[0].header.keys(),
                                input_image[0].header.values(),
                            )
                        )
                    )
                    img = input_image[2].data
                load_this_image = True
                for key, val in md_filter.items():
                    if md[key] != md_filter[key]:
                        load_this_image = False
                        # print(f'Not loading {file}, expected {key} to be {val} but it was {md[key]}')
                if load_this_image:
                    if img == None:
                        input_image = fits.open(basepath + file)
                        img = input_image[2].data
                    print(f'Loading dark for {md["EXPOSURE"]} from {file}')
                    exptime = md["EXPOSURE"]
                    self.darks[exptime] = img

    def loadSampleSpecificI0(self, file, sep="\t", column_start="Time of Day", round=1):
        """
        Load I0 values from a file and store them in the i0 DataFrame.

        Args:
            file (str): Path to the file containing I0 values.
            sep (str): Separator used in the file (default is tab).
        """
        cols = {
            "Beamline Energy": "energy",
            "EPU Polarization": "polarization",
            "Beam Current": "beam_current",
            "Photodiode": "photodiode",
            "AI 3 Izero": "i0",
        }

        header = []
        # Cycle through the file to find the end of the header
        with open(file, "r") as f:
            len_header = 0
            for line in f:
                if column_start in line:
                    break
                len_header += 1
                header.append(line.strip())
        df_load = pd.read_csv(file, sep=sep, header=(len_header - 1))
        self.i0 = df_load[list(cols)].rename(columns=cols).copy()
        self.i0.attrs["header"] = header
        self.i0["energy"] = self.i0["energy"].round(round)

    def loadSampleSpecificI1(self, file, sep="\t", column_start="Time of Day", round=1):
        """
        Load I1 values from a file and store them in the i1 DataFrame.

        Args:
            file (str): Path to the file containing I1 values.
            sep (str): Separator used in the file (default is tab).
            round (int): Number of decimal places to round the energy values (default is 1).
        """
        cols = {
            "Beamline Energy": "energy",
            "EPU Polarization": "polarization",
            "Beam Current": "beam_current",
            "Photodiode": "photodiode",
            "AI 3 Izero": "i0",
        }

        header = []
        # Cycle through the file to find the end of the header
        with open(file, "r") as f:
            len_header = 0
            for line in f:
                if column_start in line:
                    break
                len_header += 1
                header.append(line.strip())
        df_load = pd.read_csv(file, sep=sep, header=(len_header - 1))
        self.i1 = df_load[list(cols)].rename(columns=cols).copy()
        self.i1.attrs["header"] = header
        self.i1["energy"] = self.i1["energy"].round(round)

    def loadSingleImage(self, filepath, coords=None, return_q=False, **kwargs):
        """
        THIS IS A HELPER FUNCTION, mostly - should not be called directly unless you know what you are doing


        Load a single image from filepath and return a single-image, raw xarray, performing dark subtraction if so configured.

        """
        if len(kwargs.keys()) > 0:
            warnings.warn(
                f"Loader does not support features for args: {kwargs.keys()}",
                stacklevel=2,
            )
        input_image = fits.open(filepath)
        headerdict = self.normalizeMetadata(
            dict(zip(input_image[0].header.keys(), input_image[0].header.values()))
        )
        img = input_image[2].data
        # two steps in this pre-processing stage:
        #     (1) get and apply the right scalar correction term to the image
        #     (2) find and subtract the right dark
        if coords != None:
            headerdict.update(coords)

        # step 1: correction term

        if self.corr_mode == "expt":
            corr = headerdict["exposure"]  # (headerdict['AI 3 Izero']*expt)
        elif self.corr_mode == "i0":
            corr = headerdict["AI 3 Izero"]
        elif self.corr_mode == "expt+i0":
            corr = headerdict["exposure"] * headerdict["AI 3 Izero"]
        elif self.corr_mode == "user_func":
            corr = self.user_corr_func(headerdict)
        elif self.corr_mode == "old":
            corr = (
                headerdict["AI 6 BeamStop"]
                * 2.4e10
                / headerdict["Beamline Energy"]
                / headerdict["AI 3 Izero"]
            )
            # this term is a mess...  @TODO check where it comes from
        elif self.corr_mode == "als":
            # Normalize the intensity from the direct beam i0
            en = headerdict["energy"]
            ai = headerdict["AI 3 Izero"]
            exp = headerdict["exposure"]
            # Arrays from i0 scan
            if self.i0 is None:
                raise ValueError(
                    "I0 data not loaded. Please load I0 data before using 'als' correction mode."
                )
            en0 = self.i0["energy"].to_numpy(dtype=float)
            pd0 = self.i0["photodiode"].to_numpy(dtype=float)
            ai0 = self.i0["i0"].to_numpy(dtype=float)
            # Interpolate to find the corresponding i0 value for the current energy
            i0_interp = np.interp(en, en0, pd0) / np.interp(en, en0, ai0)
            corr = exp * ai * i0_interp
            # Correct the intensity for the absorption if given
            if self.i1 is not None:
                en1 = self.i1["energy"].to_numpy(dtype=float)
                pd1 = self.i1["photodiode"].to_numpy(dtype=float)
                ai1 = self.i1["i0"].to_numpy(dtype=float)
                i1_interp = (
                    np.interp(en, en1, pd1) / np.interp(en, en1, ai1)
                ) / i0_interp
                corr /= i1_interp
        else:
            corr = 1

        if corr < 0:
            warnings.warn(
                f"Correction value is negative: {corr} with headers {headerdict}.",
                stacklevel=2,
            )
            corr = abs(corr)

        # step 2: dark subtraction
        if self.dark_subtract:
            try:
                darkimg = self.darks[headerdict["EXPOSURE"]]
            except KeyError:
                warnings.warn(
                    f"Could not find a dark image with exposure time {headerdict['EXPOSURE']}.  Using zeros.",
                    stacklevel=2,
                )
                darkimg = np.zeros_like(img)
            img += self.dark_pedestal
            img = (img - darkimg) / corr
            img -= self.dark_pedestal / corr
            # img = (img - darkimg + self.dark_pedestal) / corr

        # now, match up the dims and coords
        if return_q:
            qpx = (
                2
                * np.pi
                * 60e-6
                / (headerdict["sdd"] / 1000)
                / (headerdict["wavelength"] * 1e10)
            )
            qx = (np.arange(1, img.shape[0] + 1) - headerdict["beamcenter_x"]) * qpx
            qy = (np.arange(1, img.shape[1] + 1) - headerdict["beamcenter_y"]) * qpx
            # now, match up the dims and coords
            return xr.DataArray(
                img, dims=["qy", "qx"], coords={"qy": qy, "qx": qx}, attrs=headerdict
            )
        return xr.DataArray(img, dims=["pix_x", "pix_y"], attrs=headerdict)

    def peekAtMd(self, file):
        """
        load the header/metadata without opening the corresponding image

        Args:
            file (str): fits file from which to load metadata
        """
        input_image = fits.open(file)
        headerdict = self.normalizeMetadata(
            dict(zip(input_image[0].header.keys(), input_image[0].header.values()))
        )
        return headerdict

    def normalizeMetadata(self, headerdict):
        """
        convert the local metadata terms in headerdict to standard nomenclature

        Args:
            headerdict (dict): the header returned by the file loader
        """
        headerdict["EXPOSURE"] = round(headerdict["EXPOSURE"], 4)
        headerdict["exposure"] = headerdict["EXPOSURE"] + self.exposure_offset
        headerdict["energy"] = round(headerdict["Beamline Energy"], 2)
        headerdict["polarization"] = round(headerdict["EPU Polarization"], 0) - 100
        headerdict["sam_x"] = headerdict["Sample X"]
        headerdict["sam_y"] = headerdict["Sample Y"]
        headerdict["sam_z"] = headerdict["Sample Z"]
        headerdict["sam_th"] = headerdict["Sample Theta"]
        headerdict["sampleid"] = headerdict["Sample Number"]
        headerdict["wavelength"] = 1.239842e-6 / headerdict["energy"]
        headerdict["det_x"] = round(headerdict["CCD X"], 2)
        headerdict["det_y"] = round(headerdict["CCD Y"], 2)
        headerdict["det_th"] = round(headerdict["CCD Theta"], 2)
        headerdict.update(self.constant_md)
        return headerdict
