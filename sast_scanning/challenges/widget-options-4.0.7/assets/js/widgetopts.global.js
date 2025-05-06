jQuery(function () {
  jQuery(document).on(
    "click",
    ".widgetopts-notice .notice-dismiss",
    function () {
      jQuery.ajax({
        url: widgetopts10n.ajax_url,
        type: "POST",
        data: {
          method: "delete_widgetopts_update_transient",
          action: "widgetopts_ajax_settings",
          nonce: jQuery('input[name="widgetopts-settings-nonce"]').val(),
        },
      });
    }
  );
});
