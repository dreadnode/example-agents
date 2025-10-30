<?php
/**
 * SiteOrigin Settings Module
 * Settings > Widget Options :: SiteOrigin Pagebuilder Support
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for SiteOrigin Pagebuilder Support
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */

 /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !function_exists( 'widgetopts_settings_siteorigin' ) ):
	function widgetopts_settings_siteorigin(){
	    global $widget_options; ?>
	    <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['siteorigin'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-siteorigin" data-module-id="siteorigin">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'SiteOrigin Pagebuilder Support', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Extends widget options functionality to SiteOrigin Pagebuilder Plugin.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
	                <?php if( $widget_options['siteorigin'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['siteorigin'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-editor-kitchensink"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'SiteOrigin Pagebuilder Support', 'widget-options' );?></h3>
				<p>
					<?php _e( 'This feature will enable the widget options to your widgets when you are using <a href="https://wordpress.org/plugins/siteorigin-panels/" target="_blank">Pagebuilder by SiteOrigin</a>. Easily manage and extends widget functionalities and visibility using tabbed options provided for each widgets.', 'widget-options' );?>
				</p>
				<h4><?php _e( 'Limitations', 'widget-options' );?></h4>
				<p><?php _e( ' - Visibility options tab option is not available since you are using pagebuilder.', 'widget-options' );?></p>
				<p><?php _e( ' - Custom widget ID on pagebuilder widget was removed to avoid conflicts.', 'widget-options' );?></p>
				<p>
					<?php _e( 'That\'s all! Other options will work smoothly and integrated perfectly on the plugin.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['siteorigin'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_siteorigin', 65 );
endif;
?>
