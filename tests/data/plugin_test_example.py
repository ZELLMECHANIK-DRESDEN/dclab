
def compute_some_new_features(rtdc_ds):
    """The function that does the heavy-lifting"""
    circ_per_area = rtdc_ds["circ"] / rtdc_ds["area_um"]
    circ_times_area = rtdc_ds["circ"] * rtdc_ds["area_um"]
    # returns a dictionary-like object
    return {"circ_per_area": circ_per_area, "circ_times_area": circ_times_area}


info = {
    "method": compute_some_new_features,
    "description": "This plugin will compute some possibly useless features",
    "long description": "Even longer description that "
                        "can span multiple lines",
    "feature names": ["circ_per_area", "circ_times_area"],
    "feature labels": ["Circularity per Area", "Circularity times Area"],
    "features required": ["circ", "area_um"],
    "config required": [],
    "method check required": lambda x: True,
    "priority": 1,
    "scalar feature": True,
    "version": "0.1.0",
}
