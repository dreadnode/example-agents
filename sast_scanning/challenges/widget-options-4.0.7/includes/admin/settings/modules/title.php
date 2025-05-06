<?php
/**
 * Widget Title Settings Module
 * Settings > Widget Options :: Hide Title
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       3.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Hide Widget Title
 *
 * @since 3.0
 * @global $widget_options
 * @return void
 */

/*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !function_exists( 'widgetopts_settings_title' ) ):
	function widgetopts_settings_title(){
	    global $widget_options; ?>
	    <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['hide_title'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-hide_title" data-module-id="hide_title">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Hide Title', 'widget-options' );?></h2>
				<p class="widgetopts-module-desc">
					<?php _e( 'Allows you to hide widget title on the front-end but keep them on the backend.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
	                <?php if( $widget_options['hide_title'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['hide_title'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-admin-generic"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Hide Title', 'widget-options' );?></h3>
				<p>
					<?php _e( 'Easily hide each widget title via checkbox. No need for PHP snippet to hide them per widgets.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['hide_title'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_title', 40 );
endif;
?>
