
import os
import tomlkit
from typing import Any
import hashlib

def _read_config(config_path:str):
    with open(config_path, "rb") as f:
        cfg = tomlkit.load(f)
    return cfg

class Config:
    """
    Create a config object from a toml file. Optionally, add a fallback default toml config, read from `default_path`. If `default_path` is `None`, will also check the ``CONFIG_DEFAULTS`` environment variable for a defaults filepath. 
    Profiles (toml tables) can be selected with :py:meth:`~Config.choose_profile` and deselected with :py:meth:`~Config.clear_profile`. Keys in a profile will take precedence over keys in the rest of the file and in the defaults file.
    
    
    If `"KEY"` is in both the standard config and the profile `"Profile1"`::
    
    >>> cfg = Config("config.toml",default_path="defaults.toml")
    >>> cfg["KEY"] # VAL1 
    >>> cfg.select_profile("Profile1")
    >>> cfg["KEY"] # VAL2

    If `"KEY"` is only in the standard config and not in the 'Profile1' section::
    
    >>> cfg["KEY"] # VAL1 
    >>> cfg.select_profile("Profile1")
    >>> cfg["KEY"] # VAL1

    Values can be retrieved in a few ways:: 
    
    >>> # the following are equivalent:
    >>> cfg["KEY"]
    >>> cfg("KEY")
    >>> # this allows a default value in case the key can't be found in a profile, main config, or default:
    >>> cfg.get("KEY")  # will return None if not found
    >>> cfg.get("KEY","Not found") # returns 'Not found' if not found
    >>> # this queries the default config for a key. will fail if a default config is not set:
    >>> cfg.get_default("KEY")
    
    Values can also be set. Setting a key that doesn't currently exist will add it to the config. Setting a key will change the state of the object but will not change the file unless :func:`Config.save()` is called::

    >>> cfg["KEY"] = "VALUE"  # sets in selected profile, or in main config if no profile selected
    >>> cfg["table"]["colnames"] = ["ra","dec"]  # can do nested set
    >>> cfg.set("KEY") = "VALUE"  # sets in selected profile, or in main config if no profile selected
    >>> cfg.set("KEY", profile=False) = "VALUE"  # sets in main profile, ignoring selected profile

    Can write the whole config (not just the profile, and not including the defaults) into the given file::
    
    >>> cfg.write("test.toml")
    
    Or can write to the file the config was loaded from, overwriting previous contents (does not modify defaults file)::

    >>> cfg.save()
    """
    def __init__(self,filepath:str,default_path:str|None=None,default_env_key:str="CONFIG_DEFAULTS"):
        """Create a config object from a toml file. Optionally, add a fallback default toml config, read from `default_path`. If `default_path` is `None`, will also check the CONFIG_DEFAULTS environment varaible for a defaults filepath.
         
        :param filepath: toml file to load config from
        :type filepath: str
        :param default_path: default toml file to load defaults from, defaults to None
        :type default_path: str | None, optional
        :param default_env_key: will load defaults from here if this is set and default_path is not provided, defaults to `"CONFIG_DEFAULTS"`
        :type default_env_key: str, optional
        """
        self._cfg = _read_config(filepath)
        self.selected_profile = None
        self._defaults = None
        self._filepath = filepath 
        self.selected_profile_name = None
        self._default_path = default_path
        if not self._default_path:
            self._default_path = os.getenv(default_env_key)
        if self._default_path:
            try:
                self._defaults = _read_config(self._default_path)
            except Exception as e:
                print(f"ERROR: config tried to load defaults file {self._default_path} but encountered the following: {e}")
                print("Proceeding without defaults")

    def choose_profile(self, profile_name:str):
        """Choose a profile (sub-config) to become the active profile by name 

        :param profile_name: the name of a section of this config that should be used as the active profile. Queries for keys missing from the active profile will be drawn 
        :type profile_name: str
        :return: _description_
        :rtype: _type_
        """
        self.selected_profile = self._cfg[profile_name]
        self.selected_profile_name = profile_name
        return self
    
    def clear_profile(self):
        """Deselect the currently-selected profile"""
        self.selected_profile = None
        self.selected_profile_name = None
    
    def load_defaults(self, filepath:str):
        """Load a config from ``filepath`` to use as a default config for key misses

        :param filepath: the path to the .toml file
        :type filepath: str
        """
        self._defaults = _read_config(filepath)
        self._default_path = filepath

    def write(self,fpath,trim=True):
        """Writes the whole config loaded from file (not just the profile, and not including the defaults) into the given file
        
        .. hint::
            To save a config to the same file that it was loaded from, use :py:meth:`~Config.save` 

        :param fpath: the path to the file to write to. Contents will be overwritten if it already exists
        :type fpath: str
        :param trim: whether to preserve linespacing when writing to disk, defaults to True
        :type trim: bool, optional
        """
        with open(fpath,"w") as f:
            outstr = tomlkit.dumps(self._cfg)
            if trim:
                outstr = outstr.replace("\r\n","\n")
            f.write(outstr)
    
    def save(self,trim=True):
        """Saves the whole config (not just the profile, and not including the defaults) into the file it was loaded from.
        
        .. hint::
            To save a config to any file, see :py:meth:`~Config.write` 

        :param trim: whether to preserve linespacing when writing to disk, defaults to True
        :type trim: bool, optional
        """
        self.write(self._filepath,trim=trim)

    @property
    def has_defaults(self):
        """ Whether this Config has a default config configured. 

        :rtype: bool
        """
        return self._defaults is not None
    
    def _get_default(self, key:str):
        if not self.has_defaults:
            raise AttributeError("No default configuration set!")
        return self._defaults[key]

    def get_default(self, key:str, default:Any|None=None):
        """Like :py:meth:`~Config.get`, but looks only in this Config's default layer, if configured.

        :param key: the config key to query
        :type key: str
        :param default: the default value to return if the key is not found in the config, defaults to None
        :type default: Any | None, optional
        :return: the corresponding value if one was found, or `default` if it was not
        """
        try: 
            self._get_default(key)
        except KeyError:
            return default
    
    def get(self,key:str,default:Any=None):
        """Query the Config for a key, returning a default value if not found. If a profile is selected, that profile is queried first. If the key is not present or there is no profile configured, the entire config is queried. Failing that, the default config is queried if one has been configured. If all of these miss, the value provided in ``default`` (default ``None``) will be returned.

        :param key: the key to query for
        :type key: str
        :param default: the default value to return if the key is not found, defaults to None
        :type default: Any, optional
        :return: the value associated with the key in the config, if found. otherwise, the value of the ``default`` argument.
        :rtype: Any
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def set(self,key:str,value:Any,profile:bool=True):
        """Set the value of a key in the config. If a profile is selected, the key will be set in the selected profile if ``profile`` is ``True`` (default). 
        
        .. note::
            This method does not update the underlying .toml file - see :py:meth:`~Config.save` or :py:meth:`~Config.write` to write changes made with this method to disk.

        :param key: the key to set. can be a new or existing key.
        :type key: str
        :param value: the value to assign to the key
        :type value: Any
        :param profile: whether to set the value of the key on the profile (instead of the whole config) if a profile is selected, defaults to True
        :type profile: bool, optional
        """
        if profile:
            self[key] = value
            return
        else:
            self._cfg[key] = value

    def __call__(self, index:str) -> Any:
        """Alias of :py:meth:`~Config.__getitem__`"""
        return self.__getitem__(index)

    def __getitem__(self,index:str) -> Any:
        """Get the value of key ``index`` in the config. If a profile is selected, that profile is queried first. If the key is not present in the profile or there is no profile selected, the entire config is queried. Failing that, the default config is queried if one has been configured. If all of these miss, raises ``KeyError``.

        :param index: the key to query
        :type index: str
        :raises KeyError: if the key ``index`` is not found in the config 
        :return: the value associated with the key
        :rtype: Any
        """
        if self.selected_profile:
            try:
                return self.selected_profile[index]
            except Exception:
                pass
        try:
            return self._cfg[index]    
        except Exception:
            if self.has_defaults:
                return self._get_default(index)
            raise KeyError(f"Key '{index}' not found in config {self._filepath}" + (f"or defaults {self._default_path}" if self.has_defaults else ""))
        
    def __setitem__(self,index:str,new_val:Any):
        """Set the value of a new or existing key in the config. Will set the value in the selected profile, if one has been selected. See also: :py:meth:`~Config.set`

        :param index: the key to assign a value to
        :type index: str
        :param new_val: the new value to assign
        :type new_val: Any
        """
        if self.selected_profile:
            self.selected_profile[index] = new_val
            return
        self._cfg[index] = new_val
    
    def __str__(self):
        self_str = ""
        if self.selected_profile:
            self_str = f"(Profile '{self.selected_profile_name}') "
        
        self_str += str(self._cfg)
        if self.has_defaults:
            self_str += f"\nDefaults: {self._defaults}"
        return self_str

    def __repr__(self) -> str:
        return f"Config from {self._filepath} with {f'profile {self.selected_profile_name}' if self.selected_profile_name else 'no profile'} selected and {f'defaults loaded from {self._default_path}' if self.has_defaults else 'no defaults loaded'}"

    def checksum(self):
        """Compute an md5 checksum of the string representation of this Config. useful for validating that configuration files are consistent between different pipeline runs. 

        :return: the hex digest of the checksum
        :rtype: str
        """
        return hashlib.md5(str(self).encode(),usedforsecurity=False).hexdigest()