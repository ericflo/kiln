
/* =====================================================================
   Boot pass 2: apply the full deep-link route (#page/subtab/id).
   Runs LAST on purpose — every tablist, localStorage restore, and modal
   open fn above is wired by now, so an explicit hash sub-tab overrides
   the restores (the hash always wins) and drill ids can open their
   modals. Only replaceState repairs happen here; the boot landing never
   mints a history entry, so Back still exits the dashboard.
   ===================================================================== */
applyHashRoute({ boot: true });
