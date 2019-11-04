    select ie.icustay_id, tt.starttime
      , 1 as RRT
    from mimiciii.icustays ie
    inner join mimiciii.procedureevents_mv tt
      on ie.icustay_id = tt.icustay_id
      --and tt.starttime between ie.intime and ie.intime + interval '1' day
      and itemid in
      (
          225441 -- | Hemodialysis                                      | 4-Procedures            | procedureevents_mv | Process
        , 225802 -- | Dialysis - CRRT                                   | Dialysis                | procedureevents_mv | Process
        , 225803 -- | Dialysis - CVVHD                                  | Dialysis                | procedureevents_mv | Process
        , 225805 -- | Peritoneal Dialysis                               | Dialysis                | procedureevents_mv | Process
        , 224270 -- | Dialysis Catheter                                 | Access Lines - Invasive | procedureevents_mv | Process
        , 225809 -- | Dialysis - CVVHDF                                 | Dialysis                | procedureevents_mv | Process
        , 225955 -- | Dialysis - SCUF                                   | Dialysis                | procedureevents_mv | Process
        , 225436 -- | CRRT Filter Change               | Dialysis | procedureevents_mv | Process
      )