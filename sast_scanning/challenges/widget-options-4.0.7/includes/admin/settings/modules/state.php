<?php
/**
 * User Login State Settings Module
 * Settings > Widget Options :: User Login State
 *
 * @copyright   Copyright (c) 2018, Jeffrey Carandang
 * @since       3.7
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for User Login State
 *
 * @since 3.7
 * @global $widget_options
 * @return void
 */

 /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !class_exists( 'widgetopts_settings_state' ) ){
	function widgetopts_settings_state(){
	    global $widget_options;
		//avoid issue after update
        if( !isset( $widget_options['state'] ) ){
            $widget_options['state'] = '';
        }
		?>
	    <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['state'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-state" data-module-id="state">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'User Login State', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Show widgets only for logged-in or logged-out users easily instead of display logic feature.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
	                <?php if( $widget_options['state'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['state'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-admin-users"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Logged-in or Logged-out Users Restriction', 'widget-options' );?></h3>
				<p>
					<?php _e( 'This feature will give you easier option to show specific widgets only for logged-in or logged-out users rather than using the Display Logic feature.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['state'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_state', 59 );
}
?>
