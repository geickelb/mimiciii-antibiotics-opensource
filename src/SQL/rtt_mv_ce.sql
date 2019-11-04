  select ie.icustay_id, ce.charttime
    , 1 as RRT
  from mimiciii.icustays ie
  inner join mimiciii.chartevents ce
    on ie.icustay_id = ce.icustay_id
    --and ce.charttime between ie.intime and ie.intime + interval '1' day
    and itemid in
    (
      -- Checkboxes
        225126 -- | Dialysis patient                                  | Adm History/FHPA        | chartevents        | Checkbox
      , 226118 -- | Dialysis Catheter placed in outside facility      | Access Lines - Invasive | chartevents        | Checkbox
      , 227357 -- | Dialysis Catheter Dressing Occlusive              | Access Lines - Invasive | chartevents        | Checkbox
      , 225725 -- | Dialysis Catheter Tip Cultured                    | Access Lines - Invasive | chartevents        | Checkbox
      -- Numeric values
      , 226499 -- | Hemodialysis Output                               | Dialysis                | chartevents        | Numeric
      , 224154 -- | Dialysate Rate                                    | Dialysis                | chartevents        | Numeric
      , 225810 -- | Dwell Time (Peritoneal Dialysis)                  | Dialysis                | chartevents        | Numeric
      , 227639 -- | Medication Added Amount  #2 (Peritoneal Dialysis) | Dialysis                | chartevents        | Numeric
      , 225183 -- | Current Goal                     | Dialysis | chartevents        | Numeric
      , 227438 -- | Volume not removed               | Dialysis | chartevents        | Numeric
      , 224191 -- | Hourly Patient Fluid Removal     | Dialysis | chartevents        | Numeric
      , 225806 -- | Volume In (PD)                   | Dialysis | chartevents        | Numeric
      , 225807 -- | Volume Out (PD)                  | Dialysis | chartevents        | Numeric
      , 228004 -- | Citrate (ACD-A)                  | Dialysis | chartevents        | Numeric
      , 228005 -- | PBP (Prefilter) Replacement Rate | Dialysis | chartevents        | Numeric
      , 228006 -- | Post Filter Replacement Rate     | Dialysis | chartevents        | Numeric
      , 224144 -- | Blood Flow (ml/min)              | Dialysis | chartevents        | Numeric
      , 224145 -- | Heparin Dose (per hour)          | Dialysis | chartevents        | Numeric
      , 224149 -- | Access Pressure                  | Dialysis | chartevents        | Numeric
      , 224150 -- | Filter Pressure                  | Dialysis | chartevents        | Numeric
      , 224151 -- | Effluent Pressure                | Dialysis | chartevents        | Numeric
      , 224152 -- | Return Pressure                  | Dialysis | chartevents        | Numeric
      , 224153 -- | Replacement Rate                 | Dialysis | chartevents        | Numeric
      , 224404 -- | ART Lumen Volume                 | Dialysis | chartevents        | Numeric
      , 224406 -- | VEN Lumen Volume                 | Dialysis | chartevents        | Numeric
      , 226457 -- | Ultrafiltrate Output             | Dialysis | chartevents        | Numeric
    )
    and valuenum > 0 -- also ensures it's not null