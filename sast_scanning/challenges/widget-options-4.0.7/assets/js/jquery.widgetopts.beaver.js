"use strict";

var widgetoptsBeaverModule = {
	init: function() {
		var self 	= this;
        jQuery( '.widgetopts-module-settings-container' ).hide();

		self.bindEvents();
    },

	bindEvents: function() {
		var self 	= this;
		var $body = jQuery( 'body' );

		$body.on( 'click', '.fl-builder-widgetopts-tab a', self.navClick );

	},

	navClick: function( e ) {
		e.preventDefault();
		jQuery( '.widgetopts-s-active' ).removeClass( 'widgetopts-s-active' );
		jQuery( '#fl-builder-settings-tab-widgetopts .fl-builder-settings-section' ).hide();
		jQuery( jQuery( this ).attr('href') ).show();
		jQuery( this ).addClass( 'widgetopts-s-active' );
	}
}

jQuery(document).ready(function() {
	widgetoptsBeaverModule.init();
});