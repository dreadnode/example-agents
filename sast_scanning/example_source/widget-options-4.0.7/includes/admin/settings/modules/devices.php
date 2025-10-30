<?php
/**
 * Devices Settings Module
 * Settings > Widget Options :: Devices Restriction
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Devices Visibility Options
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */

 /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !function_exists( 'widgetopts_settings_devices' ) ):
	function widgetopts_settings_devices(){
	    global $widget_options; ?>
		<li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['devices'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-devices" data-module-id="devices">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Devices Restriction', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Show or hide specific WordPress widgets on desktop, tablet and/or mobile screen sizes.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
					<?php if( $widget_options['devices'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['devices'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-smartphone"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Devices Restriction', 'widget-options' );?></h3>
				<p>
					<?php _e( 'This feature will allow you to display different sidebar widgets for each devices. You can restrict visibility on desktop, tablet and/or mobile device screen sizes at ease via checkboxes.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['devices'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_devices', 20 );
endif;
?>
