<?php
/**
 * Beaver Builder Settings Module
 * Settings > Widget Options :: Beaver Builder Pagebuilder Support
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       4.5
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Beaver Builder Pagebuilder Support
 *
 * @since 4.5
 * @global $widget_options
 * @return void
 */

  /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !class_exists( 'widgetopts_settings_beaver' ) ){
	function widgetopts_settings_beaver(){
	    global $widget_options;
		//avoid issue after update
        if( !isset( $widget_options['beaver'] ) ){
            $widget_options['beaver'] = '';
        }
		?>
	    <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['beaver'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-beaver" data-module-id="beaver">
			<div class="widgetopts-module-card-content">
				<h2><?php _e( 'Beaver Builder Support', 'widget-options' );?></h2>
				<div class="widgetopts-pro-label" style="background: transparent;color: #777;"><?php _e( 'BETA', 'widget-options' );?></div>
				<p class="widgetopts-module-desc">
					<?php _e( 'Manage module visibility & styling using Widget Options integration.', 'widget-options' );?>
				</p>

				<div class="widgetopts-module-actions hide-if-no-js">
	                <?php if( $widget_options['beaver'] == 'activate' ){ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
					<?php }else{ ?>
						<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
						<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
					<?php } ?>

				</div>
			</div>

			<?php widgetopts_modal_start( $widget_options['beaver'] ); ?>
				<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-editor-kitchensink"></span>
				<h3 class="widgetopts-modal-header"><?php _e( 'Beaver Builder Support', 'widget-options' );?></h3>
				<p>
					<?php _e( 'This feature will unlock <strong>Widget Options</strong> section on every <a href="https://www.wpbeaverbuilder.com/?utm_source=widget_options_plugin" target="_blank">Beaver Builder</a> modules that will extend functionalities and will let you control each widgets even more.', 'widget-options' );?>
				</p>
				<h4><?php _e( 'Limitations', 'widget-options' );?></h4>
				<p><?php _e( 'Widget Options feature that is already available on Beaver Builder settings will not be applied since this will be a redundant option. Thanks!', 'widget-options' );?></p>
				<div class="widgetopts-settings-section">
					<p class="widgetopts-settings-section"><?php _e( 'No additional settings available.', 'widget-options' );?></p>
				</div>
			<?php widgetopts_modal_end( $widget_options['beaver'] ); ?>

		</li>
	    <?php
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_beaver', 67 );
}
?>
