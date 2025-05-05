<?php
/**
 * Elementor Settings Module
 * Settings > Widget Options :: ACF Support
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       4.3
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Elementor Pagebuilder Support
 *
 * @since 4.3
 * @global $widget_options
 * @return void
 */

 /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !class_exists( 'widgetopts_settings_acf' ) ){
	function widgetopts_settings_acf(){
	    global $widget_options;
		//avoid issue after update
        if( !isset( $widget_options['acf'] ) ){
            $widget_options['acf'] = '';
        }
		?>
	    <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['acf'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-acf" data-module-id="acf">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Advanced Custom Fields Support', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label" style="background: transparent;color: #777;"><?php _e( 'BETA', 'widget-options' );?></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Extends widget options visibility using Advanced Custom Fields Plugin.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
	                <?php if( $widget_options['acf'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings" data-options="0"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['acf'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-editor-textcolor"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Advanced Custom Fields Support', 'widget-options' );?></h3>
				<p>
					<?php _e( 'Enabling this feature will give you more visibility options for each widget and assign condition using ACF field.', 'widget-options' );?>
				</p>
				<p class="widgetopts-settings-section">
					<?php _e( 'No additional settings available.', 'widget-options' );?>
				</p>
			<?php widgetopts_modal_end( $widget_options['acf'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_acf', 67 );
}
?>
